import torch
import torch.onnx
import torchvision
import torchvision.transforms.v2
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

        grid_cell_position_x, grid_cell_position_y = torch.meshgrid([torch.arange(0,grid.shape[3]), torch.arange(0,grid.shape[2])], indexing='ij')

        
        grid_dimensions = [torch.Tensor([grid.shape[0]]).int(),torch.Tensor([self.anchors.shape[0]]).int(), torch.Tensor([grid.shape[1]/self.anchors.shape[0]]).int(), torch.Tensor([grid.shape[2]]).int(), torch.Tensor([grid.shape[3]]).int()]
        grid = grid.view(grid_dimensions)

        anchors_tiled = self.anchors.view((self.anchors.shape[0],self.anchors.shape[1], 1, 1))
        
        x = ((grid_cell_position_x + torch.tanh(grid[:,:,0]))/grid.shape[-1]).view(grid_dimensions[0],grid_dimensions[1],1,grid_dimensions[3], grid_dimensions[4])
        y = ((grid_cell_position_y + torch.tanh(grid[:,:,1]))/grid.shape[-2]).view(grid_dimensions[0],grid_dimensions[1],1,grid_dimensions[3], grid_dimensions[4])
        w = ((anchors_tiled[:,0] * torch.exp(grid[:,:,2]))/grid.shape[-1]).view(grid_dimensions[0],grid_dimensions[1],1,grid_dimensions[3], grid_dimensions[4])
        h = ((anchors_tiled[:,1] * torch.exp(grid[:,:,3]))/grid.shape[-2]).view(grid_dimensions[0],grid_dimensions[1],1,grid_dimensions[3], grid_dimensions[4])
        
        obj = torch.sigmoid(grid[:,:,4]).view(grid_dimensions[0],grid_dimensions[1],1,grid_dimensions[3], grid_dimensions[4])

        classes = torch.sigmoid(grid[:,:,5:])

        final_boxes = torch.cat((x,y,w,h,obj,classes), dim=2)
        final_boxes = torch.permute(final_boxes, (0,1,3,4,2))
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
    

    obj_boxes = detections[:,0:4]
    obj_boxes_xyxy = torchvision.ops.box_convert(obj_boxes, "cxcywh", "xyxy")
    iou = torchvision.ops.box_iou(obj_boxes_xyxy, annotations[:,0:4])
    best_iou_ids = iou.argmax(0).tolist()

    for i in range(detections.shape[0]):
        if i in best_iou_ids:
            ann_id = best_iou_ids.index(i)
            
            batch_position_loss += torch.nn.functional.mse_loss(obj_boxes[i,0:2],annotations[ann_id,0:2],reduction="sum")
            batch_scale_loss += torch.nn.functional.mse_loss(torch.sqrt(obj_boxes[i,2:4]),torch.sqrt(annotations[ann_id,2:4]),reduction="sum")
            batch_obj_detection_loss += obj_gain * torch.nn.functional.mse_loss(detections[i,4], iou[i,ann_id])
            batch_classification_loss += torch.nn.functional.mse_loss(detections[i,5:], annotations[ann_id,4:],reduction="sum")
        else:
            batch_obj_detection_loss += no_obj_gain * torch.nn.functional.mse_loss(detections[i,4], iou[i].max())

                
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

        executor = concurrent.futures.ThreadPoolExecutor(8)
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

