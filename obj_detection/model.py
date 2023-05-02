import torch
import torchvision
import torchvision.transforms.v2
from typing import Tuple



def create_cnn_obj_detector_with_efficientnet_backbone(n_classes:int, objects_per_cell:int=1, pretrained:bool=True, train_full_model:bool=True)->torch.nn.Module:
    efficient_net = torchvision.models.efficientnet_v2_s(pretrained=pretrained)
    for param in efficient_net.parameters():
        param.requires_grad_(train_full_model)

    backbone = list(efficient_net.children())[:-2]
    cnn = torch.nn.Sequential(*backbone)

    object_data_size = 5 + n_classes # x1,y1,x2,y2,obj,classes in one_hot_encoding
    output_layer_channels = objects_per_cell*object_data_size

    n_last_layer = efficient_net.features[7][0].out_channels
    cnn.add_module("conv_out", torch.nn.Conv2d(n_last_layer, 512, 1,1))
    cnn.add_module("conv_activation", torch.nn.Sigmoid())
    cnn.add_module("output", torch.nn.Conv2d(512, output_layer_channels, 1,1))
    


    return cnn
        
def get_transforms_for_obj_detector_with_efficientnet_backbone():
    return torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(512),
        torchvision.transforms.ToTensor(),
        ])



def calc_obj_detection_loss(detections: torch.Tensor, annotations: torch.Tensor, n_masks:int, coordinates_gain:float=5., no_obj_gain:float=.5)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    grid_size = detections.size()[2:]
    n_batches = detections.size()[0]
    iou_loss = 0
    obj_detection_loss = 0
    classification_loss = 0

    batch_ratio_constant = 1./n_batches

    for batch_id in range(n_batches):
        detections_associated_with_annotations = []

        for ann_id in range(annotations.size()[1]):
            ann_box = annotations[batch_id, ann_id, 0:4].view(1,4)
            ann_class = annotations[batch_id, ann_id, 4:].view(1,-1)

            cellX = int((ann_box[0,0] + ann_box[0,2])*50/grid_size[1])
            cellY = int((ann_box[0,1] + ann_box[0,3])*50/grid_size[0])
            

            objs = detections[batch_id, :, cellY, cellX].view(n_masks,-1)
            boxes= objs[:,0:4].view(n_masks,-1)
            classes = objs[:,5:].view(n_masks,-1)

            iou = torchvision.ops.box_iou(boxes, ann_box)
            best_iou = iou.argmax(1)

            best_box = boxes[best_iou,:].view(1,4)
            best_class = classes[best_iou,:]


            iou_loss += torchvision.ops.complete_box_iou_loss(best_box, ann_box,"mean")*batch_ratio_constant
            classification_loss += (ann_class - best_class).pow(2).sum()*batch_ratio_constant
            
            detections_associated_with_annotations.append((cellY, cellX, best_iou))
        
        for cellY in range(grid_size[0]):
            for cellX in range(grid_size[1]):
                detection = detections[batch_id,:,cellY, cellX].view(n_masks,-1)
                for mask_id in range(n_masks):
                    if (cellY,cellX,mask_id) in detections_associated_with_annotations:
                        obj_detection_loss+=((1-detection[mask_id,4]).pow(2))*batch_ratio_constant
                    else:
                        obj_detection_loss+=no_obj_gain*((0-detection[mask_id,4]).pow(2))*batch_ratio_constant
            

    return coordinates_gain*iou_loss, obj_detection_loss, classification_loss


    

if __name__=="__main__":
    obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 3, True)
    input = torch.ones((1,3,512,512))
    torch.onnx.export(obj_detect, input, "obj_detect.onnx", input_names=["features"], output_names=["output"])
    print(obj_detect)





