import torch
import torch.onnx
import torchvision
from typing import Tuple, List

import concurrent.futures
import random


class YoloOutput(torch.nn.Module):

    def __init__(self, n_input_activation_maps:int, n_classes:int, anchors:List[List[int]], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_layer_channels = (5+n_classes)*len(anchors)

        self.n_classes = n_classes
        
        self.conv = torch.nn.Conv2d(n_input_activation_maps, output_layer_channels, (1, 1), 
                                    (1, 1))
        
        anchors = torch.Tensor(anchors)
        self.anchors = anchors.to(device=device)
        

    def forward(self, features: torch.Tensor):
        # Grid format -> B,C,H,W

        grid = self.conv(features)
        
        grid_cell_position_y, grid_cell_position_x = torch.meshgrid([torch.arange(grid.size(2)), torch.arange(grid.size(3))], indexing='ij')
        grid_cell_position_x = grid_cell_position_x.to(features.device)
        grid_cell_position_y = grid_cell_position_y.to(features.device)

        
        grid_dimensions = [grid.size(0),self.anchors.size(0), (5+self.n_classes), grid.size(2), grid.size(3)]
        grid = grid.reshape(grid_dimensions)

        anchors_tiled = self.anchors.reshape((self.anchors.size(0),self.anchors.size(1), 1, 1)).to(grid.device)
        
        x = ((grid_cell_position_x + torch.sigmoid(grid[:,:,0]))/grid.size(4)).unsqueeze(2)
        y = ((grid_cell_position_y + torch.sigmoid(grid[:,:,1]))/grid.size(3)).unsqueeze(2)
        w = ((anchors_tiled[:,0] * torch.exp(grid[:,:,2]))/grid.size(4)).unsqueeze(2)
        h = ((anchors_tiled[:,1] * torch.exp(grid[:,:,3]))/grid.size(3)).unsqueeze(2)
        
        obj = torch.sigmoid(grid[:,:,4]).unsqueeze(2)
        classes = torch.softmax(grid[:,:,5:],dim=2)

        # obj = grid[:,:,4].unsqueeze(2)
        # classes = grid[:,:,5:]

        final_boxes = torch.cat((x,y,w,h,obj,classes), dim=2)
        final_boxes = torch.permute(final_boxes, (0,3,4,1,2))

        if not self.training:
            final_boxes = torch.reshape(final_boxes, (grid_dimensions[0],grid_dimensions[1]*grid_dimensions[3]*grid_dimensions[4], grid_dimensions[2]))

        return final_boxes


def create_yolo_v2_model(
    n_classes: int, anchors:List[List[int]], activation=torch.nn.LeakyReLU
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

    feature_extractor.add_module(
        "prepare_features",
        torchvision.ops.Conv2dNormActivation(
            1024, 512, (3, 3), (1, 1), activation_layer=activation
        ),
    )

    output_layer = YoloOutput(512, n_classes,anchors)

    cnn.add_module("features", feature_extractor)
    cnn.add_module("output", output_layer)

    return cnn


def get_transforms_for_obj_detector():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
        ]
    )


def calc_batch_loss(detections, annotations, obj_gain, no_obj_gain):
    batch_position_loss = 0
    batch_scale_loss = 0
    batch_classification_loss = 0
    batch_obj_detection_loss = 0   
    

    obj_boxes = detections[:,:,:,0:4]

    with torch.no_grad():
        obj_boxes_xyxy = torchvision.ops.box_convert(obj_boxes, "cxcywh", "xyxy")
        ann_xyxy = torchvision.ops.box_convert(annotations[:,0:4], "cxcywh", "xyxy")
        contains_obj = torch.ones((detections.shape[0],detections.shape[1],detections.shape[2],2), device=detections.device)*(-1)

        for i in range(annotations.size(0)):
            ann_box = ann_xyxy[i]

            cellX = int((ann_box[0].item() + ann_box[2].item()) * 0.5 * detections.size(1))
            cellY = int((ann_box[1].item() + ann_box[3].item()) * 0.5 * detections.size(0))

            iou = torchvision.ops.box_iou(obj_boxes_xyxy[cellY, cellX], ann_box.view(1,-1))
            best_iou_id = iou.argmax().tolist()

            contains_obj[cellY, cellX, best_iou_id, 0] = i
            contains_obj[cellY, cellX, best_iou_id, 1] = iou[best_iou_id]

    zero = torch.zeros([1], device=detections.device)

    for i in range(detections.shape[0]):
        for j in range(detections.shape[1]):
            for k in range(detections.shape[2]):
                ann_id = contains_obj[i,j,k,0].int()
                best_iou = contains_obj[i,j,k,1]

                if ann_id >= 0:
                    batch_position_loss += torch.nn.functional.mse_loss(obj_boxes[i,j,k,0:2],annotations[ann_id,0:2],reduction="sum")
                    batch_scale_loss += torch.nn.functional.mse_loss(torch.sqrt(obj_boxes[i,j,k,2:4]),torch.sqrt(annotations[ann_id,2:4]),reduction="sum")
                    batch_obj_detection_loss += obj_gain * torch.nn.functional.mse_loss(detections[i,j,k,4], best_iou)
                    batch_classification_loss += torch.nn.functional.mse_loss(detections[i,j,k,5:], annotations[ann_id,4:],reduction="sum")
                else:
                    batch_obj_detection_loss += no_obj_gain * torch.nn.functional.mse_loss(detections[i,j,k,4].view([1]), zero)

                
    return batch_position_loss, batch_scale_loss, batch_classification_loss, batch_obj_detection_loss

def calc_obj_detection_loss(
    detections: torch.Tensor,
    annotations: torch.Tensor,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 5.0,
    no_obj_gain: float = .5,
    parallel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batches = detections.size()[0]

    position_loss = 0
    scale_loss = 0
    classification_loss = 0
    obj_detection_loss = 0

    

    if parallel:

        executor = concurrent.futures.ThreadPoolExecutor(n_batches)
        batch_processing = []

        for batch_id in range(n_batches):

            batch_process = executor.submit(calc_batch_loss, detections[batch_id], annotations[batch_id], obj_gain, no_obj_gain)
            batch_processing.append(batch_process)

        for batch_process in batch_processing:
            
            batch_position_loss, batch_scale_loss, batch_classification_loss, batch_obj_detection_loss = batch_process.result()

            position_loss += batch_position_loss
            scale_loss += batch_scale_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss
    else:

        for batch_id in range(n_batches):
            batch_position_loss, batch_scale_loss, batch_classification_loss, batch_obj_detection_loss = calc_batch_loss(detections[batch_id], annotations[batch_id], obj_gain, no_obj_gain)
            
            position_loss += batch_position_loss
            scale_loss += batch_scale_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss

    return (
        (coordinates_gain) * position_loss,
        (coordinates_gain) * scale_loss,
        obj_detection_loss,
        (classification_gain) * classification_loss,
    )


if __name__ == "__main__":
    # obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 1, True)
    obj_detect = create_yolo_v2_model(2, [[1.5686,1.5720], [4.1971,4.4082], [7.4817,7.1439], [11.1631,8.4289], [12.9989,12.5632]])

    # obj_detect.eval()

    # dynamic_params = {"features":{0:"batch_size", 2:"image_height", 3:"image_width"}, "output":{0:"batch_size",1:"n_boxes"}}
    input_sample = torch.ones((1, 3, 512, 512))
    model_file_name = f"obj_det.onnx"
    torch.onnx.export(
        obj_detect,
        input_sample,
        model_file_name,
        input_names=["features"],
        output_names=["output"],
        # dynamic_axes=dynamic_params
    )
    print(obj_detect)

