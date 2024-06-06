import torch
from typing import List, Tuple, Callable
import torchvision
import concurrent.futures
import mlflow
import mlflow.pytorch

class YoloOutput(torch.nn.Module):

    def __init__(self, n_input_activation_maps:int, n_classes:int, anchors:List[List[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_layer_channels = (5+n_classes)*len(anchors)        
        self.conv = torch.nn.Conv2d(n_input_activation_maps, output_layer_channels, (1, 1), (1, 1))
        self.n_classes = torch.tensor(n_classes)
        self.anchors = torch.Tensor(anchors)
        

    def forward(self, features: torch.Tensor):
        # Grid format -> B,C,H,W

        grid = self.conv(features)
        
        with torch.no_grad():
            grid_cell_position_y, grid_cell_position_x = torch.meshgrid([torch.arange(grid.size(2)), torch.arange(grid.size(3))], indexing='ij')
            grid_cell_position_y = grid_cell_position_y.to(features.device)
            grid_cell_position_x = grid_cell_position_x.to(features.device)
            grid_dimensions = [grid.size(0),self.anchors.size(0), (5+self.n_classes), grid.size(2), grid.size(3)]
        
        grid = grid.reshape(grid_dimensions).to(features.device)
        anchors_tiled = self.anchors.reshape((self.anchors.size(0),self.anchors.size(1), 1, 1)).to(features.device)
        
        x = ((grid_cell_position_x + torch.sigmoid(grid[:,:,0]))/grid.size(4)).unsqueeze(2)
        y = ((grid_cell_position_y + torch.sigmoid(grid[:,:,1]))/grid.size(3)).unsqueeze(2)
        w = ((anchors_tiled[:,0] * torch.exp(grid[:,:,2]))/grid.size(4)).unsqueeze(2)
        h = ((anchors_tiled[:,1] * torch.exp(grid[:,:,3]))/grid.size(3)).unsqueeze(2)
        
        obj = torch.sigmoid(grid[:,:,4]).unsqueeze(2)
        classes = torch.sigmoid(grid[:,:,5:])

        final_boxes = torch.cat((x,y,w,h,obj,classes), dim=2)

        return final_boxes
    



class YoloV2(torch.nn.Module):
    def __init__(self, n_input_channels:int, n_classes:int, anchors:List[List[int]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torchvision.ops.Conv2dNormActivation(n_input_channels, 16, (3, 3), (1, 1))
        self.conv2 = torchvision.ops.Conv2dNormActivation(16, 32, (3, 3), (1, 1))
        self.conv3 = torchvision.ops.Conv2dNormActivation(32, 64, (3, 3), (1, 1))
        self.conv4 = torchvision.ops.Conv2dNormActivation(64, 128, (3, 3), (1, 1))
        self.conv5 = torchvision.ops.Conv2dNormActivation(128, 256, (3, 3), (1, 1))
        self.conv6 = torchvision.ops.Conv2dNormActivation(256, 512, (3, 3), (1, 1))
        self.conv7 = torchvision.ops.Conv2dNormActivation(512, 1024, (3, 3), (1, 1))

        self.conv8 = torchvision.ops.Conv2dNormActivation(1024, 512, (3, 3), (1, 1))
        self.output = YoloOutput(512, n_classes, anchors)

        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=1)
    
    def forward(self, image):
        i = image.float()
        x = self.conv1(i)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.pool1(x)

        l1 = self.conv5(x)
        x = self.pool1(l1)

        x = self.conv6(x)
        x = self.pool2(x)

        head_result = self.conv7(x)

        x = self.conv8(head_result)
        o1 = self.output(x)

        o1 = torch.permute(o1, (0,1,3,4,2))

        return o1



    def save_model(self, name: str = "yolov2", input_size = (416,416), device = None):
        
        mlflow.pytorch.log_model(self, name, extra_files=[__file__])
        
        input_sample = torch.ones((1, 3, input_size[1], input_size[0]))
        if device is not None:
            input_sample = input_sample.to(device)
        model_file_name = f"{name}.onnx"
        # dynamic_params = {"features":{0:"batch_size", 2:"image_height", 3:"image_width"}, 
        #                   "output1":{0:"batch_size",3:"grid_y",4:"grid_x"},
        #                   "output2":{0:"batch_size",3:"grid_y",4:"grid_x"}}
        torch.onnx.export(
            self,
            input_sample,
            model_file_name,
            input_names=["features"],
            output_names=["output"],
            opset_version=11
            # dynamic_axes=dynamic_params
        )

        mlflow.log_artifact(model_file_name, f"onnx/{model_file_name}")


def calc_batch_loss(detections:torch.Tensor, annotations, class_loss, obj_gain, no_obj_gain):
    batch_position_loss = 0
    batch_classification_loss = 0
    batch_obj_detection_with_obj_loss = 0   
    batch_obj_detection_without_obj_loss = 0
    
    obj_boxes = detections[:,:,:,0:4]
    obj_boxes_xyxy = torchvision.ops.box_convert(obj_boxes, "cxcywh", "xyxy")
    
    zero = torch.zeros([1], device=detections.device,requires_grad=False)[0]

    if annotations["boxes"].shape[0] > 0:
        
        ann_boxes = annotations["boxes"].to(detections.device)
        ann_classes = annotations["labels"].to(detections.device)
        
        ann_xyxy = torchvision.ops.box_convert(ann_boxes, "cxcywh", "xyxy").detach()
        contains_obj = {}

        for i in range(ann_xyxy.size(0)):
            ann_box = ann_xyxy[i]

            cellX = int((ann_box[0].item() + ann_box[2].item()) * 0.5 * detections.size(2))
            cellY = int((ann_box[1].item() + ann_box[3].item()) * 0.5 * detections.size(1))

            iou = torchvision.ops.box_iou(obj_boxes_xyxy[:,cellY, cellX,:], ann_box.view(1,-1))
            best_iou_id = iou.argmax().tolist()

            contains_obj[(best_iou_id, cellY, cellX)] = (i,iou[best_iou_id])

        for i in range(detections.shape[0]):
            for j in range(detections.shape[1]):
                for k in range(detections.shape[2]):
                    pos = (i,j,k)
                    
                    if pos in contains_obj:
                        ann_id, best_iou = contains_obj[pos]
                        batch_position_loss += 1-best_iou #torchvision.ops.distance_box_iou_loss(obj_boxes_xyxy[i,j,k], ann_xyxy[ann_id])
                        batch_obj_detection_with_obj_loss += obj_gain * torch.nn.functional.mse_loss(detections[i,j,k,4], best_iou.detach())
                        batch_classification_loss += class_loss(detections[i,j,k,5:], ann_classes[ann_id])
                    else:
                        batch_obj_detection_without_obj_loss += no_obj_gain * torch.nn.functional.mse_loss(detections[i,j,k,4], zero)

        # batch_position_loss = batch_position_loss / annotations["boxes"].shape[0]
        # batch_classification_loss = batch_classification_loss / annotations["boxes"].shape[0]
        # batch_obj_detection_with_obj_loss = batch_obj_detection_with_obj_loss / annotations["boxes"].shape[0]
        # batch_obj_detection_without_obj_loss = batch_obj_detection_without_obj_loss / ((detections.shape[0]*detections.shape[1]*detections.shape[2]) - annotations["boxes"].shape[0])

    else:
        for i in range(detections.shape[0]):
            for j in range(detections.shape[1]):
                for k in range(detections.shape[2]):
                    batch_obj_detection_without_obj_loss += no_obj_gain * torch.nn.functional.mse_loss(detections[i,j,k,4], zero)
        
        # batch_obj_detection_without_obj_loss = batch_obj_detection_without_obj_loss / (detections.shape[0]*detections.shape[1]*detections.shape[2])
    
    
    return batch_position_loss, batch_classification_loss, batch_obj_detection_with_obj_loss + batch_obj_detection_without_obj_loss


def calc_obj_detection_loss(
    detections: torch.Tensor,
    annotations: torch.Tensor,
    class_loss: Callable = torch.nn.functional.binary_cross_entropy,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 5.0,
    no_obj_gain: float = .5,
    parallel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batches = detections.size()[0]

    position_loss = 0
    scale_loss = 0
    classification_loss = 0
    obj_detection_loss = 0
    
    if parallel:

        executor = concurrent.futures.ThreadPoolExecutor(n_batches)
        batch_processing = []

        for batch_id in range(n_batches):

            batch_process = executor.submit(calc_batch_loss, detections[batch_id], annotations[batch_id], class_loss, obj_gain, no_obj_gain)
            batch_processing.append(batch_process)

        for batch_process in batch_processing:
            
            batch_position_loss, batch_classification_loss, batch_obj_detection_loss = batch_process.result()

            position_loss += batch_position_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss
    else:

        for batch_id in range(n_batches):
            
            batch_position_loss, batch_classification_loss, batch_obj_detection_loss = calc_batch_loss(detections[batch_id], annotations[batch_id], class_loss, obj_gain, no_obj_gain)
            
            position_loss += batch_position_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss

    return (
        (coordinates_gain * position_loss)/n_batches,
        (obj_detection_loss)/n_batches,
        (classification_gain * classification_loss)/n_batches
    )


if __name__ == "__main__":
    device = torch.device("cuda")
    model = YoloV2(3, 2, [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]]).to(device)

    input_sample = torch.ones((1,3, 416,416))*0.5
    input_sample = input_sample.to(device)

    import time

    mean = 0

    for i in range(10):

        start = time.time()

        output = model(input_sample)

        end = time.time()

        v = (end - start)
        mean += v
        print(v)
    
        time.sleep(1)

    print()
    print(mean/10)

    model.save_model(device=device)
    

    ...
