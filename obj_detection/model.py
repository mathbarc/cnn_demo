import torch
import torch.onnx
import torchvision
import torchvision.transforms.v2
from typing import Tuple

import concurrent.futures
import random

def create_output_layer(
    n_inputs, n_classes, objects_per_cell, activation=torch.nn.SiLU
):
    object_data_size = 5 + n_classes  # x1,y1,x2,y2,obj,classes in one_hot_encoding
    output_layer_channels = objects_per_cell * object_data_size

    output_layer = torch.nn.Sequential()

    output_layer.add_module(
        "prepare_features",
        torchvision.ops.Conv2dNormActivation(
            n_inputs, 512, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    output_layer.add_module(
        "gen_output_grid",
        torch.nn.Conv2d(512, output_layer_channels, (1, 1), (1, 1)),
    )

    return output_layer

class RegionOutputLayer(torch.nn.Module):

    def __init__(self, n_objects_per_cell, n_inputs, n_classes, activation=torch.nn.SiLU, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_objects_per_cell = n_objects_per_cell
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        self.layers = create_output_layer(n_inputs, n_classes, n_objects_per_cell, activation)
        self.anchors = torch.Tensor([[47, 43],  [94,105], [210,207], [361,283], [442,425]]).to(device=device)*(1./512.)

    def forward(self, input):

        detections = self.layers(input).permute((0,2,3,1))

        boxes = None
        objectiviness = None
        classes = None


        for batch_i in range(detections.shape[0]):
            batch_boxes = None
            batch_objectiviness = None
            batch_classes = None

            for y in range(detections.shape[1]):
                for x in range(detections.shape[2]):
            
                    objs = detections[batch_i,y,x, :].view((self.n_objects_per_cell,-1))
                    
                    bx = ((x + torch.nn.functional.sigmoid(objs[:,0]))/(detections.shape[2])).view(self.n_objects_per_cell,1)
                    by = ((y + torch.nn.functional.sigmoid(objs[:,1]))/(detections.shape[1])).view(self.n_objects_per_cell,1)
                    bw = (self.anchors[:,0]*torch.exp(objs[:,2])).view(self.n_objects_per_cell,1)
                    bh = (self.anchors[:,1]*torch.exp(objs[:,3])).view(self.n_objects_per_cell,1)

                    box = torch.cat((bx, by, bw, bh),1)
                    if batch_boxes is None:
                        batch_boxes = box
                    else:
                        batch_boxes = torch.cat((batch_boxes, box))

                    objness = torch.nn.functional.sigmoid(objs[:,4])
                    if batch_objectiviness is None:
                        batch_objectiviness = objness
                    else:
                        batch_objectiviness = torch.cat((batch_objectiviness, objness))

                    obj_classes = torch.nn.functional.softmax(objs[:,5:], dim=1)
                    if batch_classes is None:
                        batch_classes = obj_classes
                    else:
                        batch_classes = torch.cat((batch_classes, obj_classes))
            
            batch_boxes = batch_boxes.view((1,detections.shape[1],detections.shape[2], self.n_objects_per_cell, batch_boxes.shape[1]))
            batch_objectiviness = batch_objectiviness.view((1,detections.shape[1],detections.shape[2], self.n_objects_per_cell))
            batch_classes = batch_classes.view((1,detections.shape[1],detections.shape[2], self.n_objects_per_cell, batch_classes.shape[1]))

            if boxes is None:
                boxes = batch_boxes
                objectiviness = batch_objectiviness
                classes = batch_classes
            else:
                boxes = torch.cat((boxes, batch_boxes))
                objectiviness = torch.cat((objectiviness, batch_objectiviness))
                classes = torch.cat((classes, batch_classes))


        return boxes, objectiviness, classes

def apply_activation_to_objects_from_output(detections, n_objects_per_cell, anchors=torch.Tensor([[47, 43],  [94,105], [210,207], [361,283], [442,425]])):

    anchors = anchors.to(detections.device)

    objs = detections.view((detections.shape[0],detections.shape[1],n_objects_per_cell,-1))
    boxes = None
    objectiviness = None
    classes = None
    
    for x in range(detections.shape[1]):
        for y in range(detections.shape[0]):
            
            objs = detections[y,x, :].view((n_objects_per_cell,-1))

            bx = ((x + torch.nn.functional.sigmoid(objs[:,0]))/(detections.shape[2])).view(n_objects_per_cell,1)
            by = ((y + torch.nn.functional.sigmoid(objs[:,1]))/(detections.shape[1])).view(n_objects_per_cell,1)
            bw = (anchors[:,0]*torch.exp(objs[:,2])/(512)).view(n_objects_per_cell,1)
            bh = (anchors[:,1]*torch.exp(objs[:,3])/(512)).view(n_objects_per_cell,1)

            box = torch.cat((bx, by, bw, bh),1)
            if boxes is None:
                boxes = box
            else:
                boxes = torch.cat((boxes, box))

            objness = torch.nn.functional.sigmoid(objs[:,4])
            if objectiviness is None:
                objectiviness = objness
            else:
                objectiviness = torch.cat((objectiviness, objness))

            obj_classes = torch.nn.functional.softmax(objs[:,5:])
            if classes is None:
                classes = obj_classes
            else:
                classes = torch.cat((classes, obj_classes))
    
    boxes = boxes.view((detections.shape[0],detections.shape[1], n_objects_per_cell, boxes.shape[1]))
    objectiviness = objectiviness.view((detections.shape[0],detections.shape[1], n_objects_per_cell))
    classes = classes.view((detections.shape[0],detections.shape[1], n_objects_per_cell, classes.shape[1]))

    return boxes, objectiviness, classes


def create_cnn_obj_detector_with_efficientnet_backbone(
    n_classes: int,
    objects_per_cell: int = 1,
    pretrained: bool = True,
    train_full_model: bool = True,
) -> torch.nn.Module:
    efficient_net = torchvision.models.efficientnet_v2_s(pretrained=pretrained)
    for param in efficient_net.parameters():
        param.requires_grad_(train_full_model)

    backbone = list(efficient_net.children())[:-2]
    cnn = torch.nn.Sequential(*backbone)

    n_last_layer = efficient_net.features[7][0].out_channels

    output_layer = create_output_layer(n_last_layer, n_classes, objects_per_cell)

    cnn.add_module("output", output_layer)

    return cnn


def create_yolo_v2_model(
    n_classes: int, objects_per_cell: int = 1, activation=torch.nn.LeakyReLU
):
    cnn = torch.nn.Sequential()

    feature_extractor = torch.nn.Sequential()

    feature_extractor.add_module(
        "0_conv",
        torchvision.ops.Conv2dNormActivation(
            3, 16, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("0_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "1_conv",
        torchvision.ops.Conv2dNormActivation(
            16, 32, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("1_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "2_conv",
        torchvision.ops.Conv2dNormActivation(
            32, 64, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("2_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "3_conv",
        torchvision.ops.Conv2dNormActivation(
            64, 128, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("3_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "4_conv",
        torchvision.ops.Conv2dNormActivation(
            128, 256, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("4_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "5_conv",
        torchvision.ops.Conv2dNormActivation(
            256, 512, (3, 3), (1, 1), activation_layer=activation
        ),
    )
    feature_extractor.add_module("5_pool", torch.nn.MaxPool2d((2, 2), (1, 1)))
    feature_extractor.add_module(
        "6_conv",
        torchvision.ops.Conv2dNormActivation(
            512, 1024, (3, 3), (1, 1), activation_layer=activation
        ),
    )

    # output_layer = RegionOutputLayer(objects_per_cell, 1024, n_classes, activation)
    output_layer = create_output_layer(1024,n_classes, objects_per_cell, activation)

    cnn.add_module("features", feature_extractor)
    cnn.add_module("output", output_layer)

    return cnn


def get_transforms_for_obj_detector():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
        ]
    )





def calc_batch_loss(detections, annotations, n_objects_per_cell, obj_gain, no_obj_gain):
    batch_iou_loss = 0
    batch_classification_loss = 0
    batch_obj_detection_loss = 0

    boxes, objectiviness, classes = apply_activation_to_objects_from_output(detections, n_objects_per_cell)


    detections_associated_with_annotations = torch.zeros(
        (objectiviness.shape),requires_grad=False
    ).to(objectiviness.device)

    grid_size = boxes.shape[0:3]

    for ann_id in range(annotations.shape[0]):
        ann_box = annotations[ann_id, 0:4]
        ann_class = annotations[ann_id, 4:]

        cellX = int((ann_box[0].item() + ann_box[2].item()) * 0.5 * grid_size[1])
        cellY = int((ann_box[1].item() + ann_box[3].item()) * 0.5 * grid_size[0])

        obj_boxes = torchvision.ops.box_convert(boxes[cellY, cellX, :, :], "cxcywh", "xyxy")
        
        iou = torchvision.ops.box_iou(obj_boxes, ann_box.view(1,-1))
        best_iou_id = iou.argmax().item()

        while detections_associated_with_annotations[cellY, cellX, best_iou_id].item() == 1 and iou.sum()>0:
            iou[best_iou_id] = 0
            best_iou_id = iou.argmax().item()

        if iou[best_iou_id] == 0:
            best_iou_id = random.randint(0,obj_boxes.shape[0]-1)

        
        batch_iou_loss += torch.nn.functional.smooth_l1_loss(obj_boxes[best_iou_id], ann_box)

        batch_classification_loss += torch.nn.functional.cross_entropy(classes[cellY, cellX, best_iou_id, :],ann_class.argmax())

        batch_obj_detection_loss += obj_gain*torch.nn.functional.binary_cross_entropy(objectiviness[cellY, cellX, best_iou_id].view((1)), iou[best_iou_id])

        detections_associated_with_annotations[cellY, cellX, best_iou_id] = 1
    
    if no_obj_gain > 0:
        
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):    
                for box_id in range(grid_size[2]):
                    target = detections_associated_with_annotations[y,x,box_id].view(1)
                    if target.item() == 0:
                        batch_obj_detection_loss += no_obj_gain*torch.nn.functional.binary_cross_entropy(objectiviness[cellY, cellX, box_id].view((1)), target)
                
    return batch_iou_loss, batch_classification_loss, batch_obj_detection_loss

def calc_obj_detection_loss(
    detections: torch.Tensor,
    annotations: torch.Tensor,
    n_objects_per_cell: int,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 5.0,
    no_obj_gain: float = .5,
    parallel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batches = detections.size()[0]

    iou_loss = 0
    classification_loss = 0
    obj_detection_loss = 0

    detections = detections.permute(0,2,3,1)

    if parallel:

        executor = concurrent.futures.ThreadPoolExecutor(8)
        batch_processing = []

        for batch_id in range(n_batches):

            batch_process = executor.submit(calc_batch_loss, detections[batch_id], annotations[batch_id], n_objects_per_cell, obj_gain, no_obj_gain)
            batch_processing.append(batch_process)

        for batch_process in batch_processing:
            
            batch_iou_loss, batch_classification_loss, batch_obj_detection_loss = batch_process.result()

            iou_loss += batch_iou_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss
    else:

        for batch_id in range(n_batches):
            batch_iou_loss, batch_classification_loss, batch_obj_detection_loss = calc_batch_loss(detections[batch_id], annotations[batch_id], n_objects_per_cell, obj_gain, no_obj_gain)
            
            iou_loss += batch_iou_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss

    return (
        (coordinates_gain) * iou_loss,
        obj_detection_loss,
        (classification_gain) * classification_loss,
    )


if __name__ == "__main__":
    # obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 1, True)
    obj_detect = create_yolo_v2_model(2, 5)

    input = torch.ones((1, 3, 512, 512))
    torch.onnx.export(
        obj_detect,
        input,
        "obj_detect.onnx",
        input_names=["features"],
        output_names=["output"],
    )
    print(obj_detect)

