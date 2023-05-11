import torch
import torch.onnx
import torchvision
import torchvision.transforms.v2
from typing import Tuple

import concurrent.futures


def create_output_layer(
    n_inputs, n_classes, objects_per_cell, activation=torch.nn.SiLU
):
    object_data_size = 5 + n_classes  # x1,y1,x2,y2,obj,classes in one_hot_encoding
    output_layer_channels = objects_per_cell * object_data_size

    output_layer = torch.nn.Sequential()

    output_layer.add_module(
        "prepare_features",
        torchvision.ops.Conv2dNormActivation(
            n_inputs, 512, (1, 1), (1, 1), activation_layer=activation
        ),
    )
    output_layer.add_module(
        "gen_output_grid",
        torch.nn.Conv2d(512, output_layer_channels, (1, 1), (1, 1)),
    )

    return output_layer


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
    feature_extractor.add_module("5_pool", torch.nn.MaxPool2d((2, 2), (2, 2)))
    feature_extractor.add_module(
        "6_conv",
        torchvision.ops.Conv2dNormActivation(
            512, 1024, (3, 3), (1, 1), activation_layer=activation
        ),
    )

    output_layer = create_output_layer(1024, n_classes, objects_per_cell, activation)

    cnn.add_module("features", feature_extractor)
    cnn.add_module("output", output_layer)

    return cnn


def get_transforms_for_obj_detector():
    return torchvision.transforms.Compose(
        [torchvision.transforms.Resize(416), torchvision.transforms.ToTensor()]
    )


def calc_batch_loss(detections, annotations, n_objects_per_cell, obj_gain, no_obj_gain):
    batch_iou_loss = 0
    batch_classification_loss = 0
    batch_obj_detection_loss = 0

    detections_associated_with_annotations = torch.zeros(
        (detections.shape[1], detections.shape[2], n_objects_per_cell),requires_grad=False
    ).to(detections.device)

    grid_size = detections.shape[:2]

    for ann_id in range(annotations.shape[0]):
        ann_box = annotations[ann_id, 0:4]
        ann_class = annotations[ann_id, 4:]

        cellX = int((ann_box[0] + ann_box[2]) * 0.5 * grid_size[1])
        cellY = int((ann_box[1] + ann_box[3]) * 0.5 * grid_size[0])

        objs = detections[cellY, cellX, :].view(n_objects_per_cell, -1)
        boxes = torchvision.ops.box_convert(torch.nn.functional.sigmoid(objs[:, 0:4]), "xywh", "xyxy")
        classes = torch.nn.functional.softmax(objs[:, 5:],dim=1)

        iou = torchvision.ops.box_iou(boxes, ann_box.view(1,-1))
        best_iou_id = iou.argmax().item()
        
        batch_iou_loss += torchvision.ops.complete_box_iou_loss(boxes[best_iou_id], ann_box)

        batch_classification_loss += torch.nn.functional.binary_cross_entropy(classes[best_iou_id, :],ann_class)

        detections_associated_with_annotations[cellY, cellX, best_iou_id] = iou[best_iou_id].item()

    all_objects = detections.reshape(
        grid_size[0] * grid_size[1] * n_objects_per_cell, -1
    )

    for cellY in range(grid_size[0]):
        for cellX in range(grid_size[1]):
            start = (cellY * grid_size[1] + cellX) * n_objects_per_cell
            end = (cellY * grid_size[1] + cellX + 1) * n_objects_per_cell
            detection = all_objects[start:end, :]
            for object_id in range(n_objects_per_cell):
                pred = torch.nn.functional.sigmoid(detection[object_id, 4])
                target = detections_associated_with_annotations[cellY, cellX, object_id]
                if target:
                    batch_obj_detection_loss += (
                            obj_gain * torch.nn.functional.binary_cross_entropy(pred,target)
                        )
                else:
                    batch_obj_detection_loss += (
                            no_obj_gain * torch.nn.functional.binary_cross_entropy(pred,target)
                        )
                
    return batch_iou_loss, batch_classification_loss, batch_obj_detection_loss

def calc_obj_detection_loss(
    detections: torch.Tensor,
    annotations: torch.Tensor,
    n_objects_per_cell: int,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 5.0,
    no_obj_gain: float = .5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batches = detections.size()[0]

    iou_loss = 0
    classification_loss = 0
    obj_detection_loss = 0

    detections = detections.permute((0, 2, 3, 1))

    n_annotations = annotations.shape[1]

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

    return (
        (coordinates_gain/n_annotations) * iou_loss,
        obj_detection_loss/n_annotations,
        (classification_gain/n_annotations) * classification_loss,
    )


if __name__ == "__main__":
    # obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 1, True)
    obj_detect = create_yolo_v2_model(2, 5)

    input = torch.ones((1, 3, 416, 416))
    torch.onnx.export(
        obj_detect,
        input,
        "obj_detect.onnx",
        input_names=["features"],
        output_names=["output"],
    )
    print(obj_detect)

