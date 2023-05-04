import torch
import torch.onnx
import torchvision
import torchvision.transforms.v2
from typing import Tuple

def create_output_layer(n_inputs, n_classes, objects_per_cell):
    object_data_size = 5 + n_classes # x1,y1,x2,y2,obj,classes in one_hot_encoding
    output_layer_channels = objects_per_cell*object_data_size

    output_layer = torch.nn.Sequential()
    
    output_layer.add_module("prepare_features",torch.nn.Conv2d(n_inputs, 512, (1,1), (1,1)))
    output_layer.add_module("prepare_features_activition",torch.nn.SiLU())
    output_layer.add_module("prepare_features_batchnorm",torch.nn.BatchNorm2d(512))
    output_layer.add_module("gen_output_grid",torch.nn.Conv2d(512, output_layer_channels, (1,1), (1,1)))

    return output_layer

def create_cnn_obj_detector_with_efficientnet_backbone(n_classes:int, objects_per_cell:int=1, pretrained:bool=True, train_full_model:bool=True)->torch.nn.Module:
    efficient_net = torchvision.models.efficientnet_v2_s(pretrained=pretrained)
    for param in efficient_net.parameters():
        param.requires_grad_(train_full_model)

    backbone = list(efficient_net.children())[:-2]
    cnn = torch.nn.Sequential(*backbone)

    n_last_layer = efficient_net.features[7][0].out_channels

    output_layer = create_output_layer(n_last_layer, n_classes, objects_per_cell)

    cnn.add_module("output", output_layer)

    return cnn

def create_potato_model(n_classes:int, objects_per_cell:int=1):
    cnn = torch.nn.Sequential()

    feature_extractor = torch.nn.Sequential()
    feature_extractor.add_module("0_conv",torch.nn.Conv2d(3, 8,(3,3),(1,1)))
    feature_extractor.add_module("0_act",torch.nn.SiLU())
    feature_extractor.add_module("0_batch_norm",torch.nn.BatchNorm2d(8))
    feature_extractor.add_module("0_pool",torch.nn.MaxPool2d((2,2),(2,2)))

    feature_extractor.add_module("1_conv",torch.nn.Conv2d(8, 16,(3,3),(1,1)))
    feature_extractor.add_module("1_act",torch.nn.SiLU())
    feature_extractor.add_module("1_batch_norm",torch.nn.BatchNorm2d(16))
    feature_extractor.add_module("1_pool",torch.nn.MaxPool2d((2,2),(2,2)))

    feature_extractor.add_module("2_conv",torch.nn.Conv2d(16, 32,(3,3),(1,1)))
    feature_extractor.add_module("2_act",torch.nn.SiLU())
    feature_extractor.add_module("2_batch_norm",torch.nn.BatchNorm2d(32))
    feature_extractor.add_module("2_pool",torch.nn.MaxPool2d((2,2),(2,2)))

    feature_extractor.add_module("3_conv",torch.nn.Conv2d(32, 64,(3,3),(1,1)))
    feature_extractor.add_module("3_act",torch.nn.SiLU())
    feature_extractor.add_module("3_batch_norm",torch.nn.BatchNorm2d(64))
    feature_extractor.add_module("3_pool",torch.nn.MaxPool2d((2,2),(2,2)))

    feature_extractor.add_module("4_conv",torch.nn.Conv2d(64, 256,(3,3),(1,1)))
    feature_extractor.add_module("4_act",torch.nn.SiLU())
    feature_extractor.add_module("4_batch_norm",torch.nn.BatchNorm2d(256))
    feature_extractor.add_module("4_pool",torch.nn.MaxPool2d((4,4),(4,4)))

    feature_extractor.add_module("out",torch.nn.Conv2d(256,1024,(3,3),(1,1)))
    feature_extractor.add_module("out_act",torch.nn.SiLU())
    feature_extractor.add_module("out_batch_norm",torch.nn.BatchNorm2d(1024))

    output_layer = create_output_layer(1024,n_classes, objects_per_cell)

    cnn.add_module("features", feature_extractor)
    cnn.add_module("output", output_layer)

    return cnn


        
def get_transforms_for_obj_detector():
    return torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(512),
        torchvision.transforms.ToTensor(),
        ])



def calc_obj_detection_loss(detections: torch.Tensor, annotations: torch.Tensor, n_objects_per_cell:int, coordinates_gain:float=5., no_obj_gain:float=.5)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    grid_size = detections.size()[2:]
    n_batches = detections.size()[0]
    iou_loss = 0
    obj_detection_loss = 0
    classification_loss = 0

    n_annotations = annotations.size()[1]
    n_all_annotations = n_batches*n_annotations

    detections = detections.permute((0,2,3,1))

    for batch_id in range(n_batches):
        detections_associated_with_annotations = []

        batch_iou_loss = 0
        batch_obj_detection_loss = 0
        batch_classification_loss = 0

        for ann_id in range(n_annotations):
            ann_box = annotations[batch_id, ann_id, 0:4].view(1,4)
            ann_class = annotations[batch_id, ann_id, 4:]

            cellX = int((ann_box[0,0] + ann_box[0,2])*0.5*grid_size[1])
            cellY = int((ann_box[0,1] + ann_box[0,3])*0.5*grid_size[0])
            

            objs = detections[batch_id, cellY, cellX, :].view(n_objects_per_cell,-1)
            boxes= objs[:,0:4].view(n_objects_per_cell,-1)
            classes = objs[:,5:].view(n_objects_per_cell,-1)

            iou = torchvision.ops.box_iou(boxes, ann_box)
            best_iou = iou.argmax().item()

            best_box = boxes[best_iou,:].view(1,4)
            best_class = classes[best_iou,:]


            batch_iou_loss += torch.nn.functional.mse_loss(ann_box,best_box,reduction="sum")
            batch_classification_loss += torch.nn.functional.mse_loss(ann_class,best_class,reduction="sum")
            
            detections_associated_with_annotations.append((cellY, cellX, best_iou))
        
        for cellY in range(grid_size[0]):
            for cellX in range(grid_size[1]):
                detection = detections[batch_id,cellY, cellX,:].view(n_objects_per_cell,-1)
                for mask_id in range(n_objects_per_cell):
                    pred = detection[mask_id,4].view(1)
                    if (cellY,cellX,mask_id) in detections_associated_with_annotations:
                        target = torch.ones(pred.size()).to(pred.device)
                        batch_obj_detection_loss+=torch.nn.functional.mse_loss(target,pred,reduction="sum")
                    else:
                        target = torch.zeros(pred.size()).to(pred.device)
                        batch_obj_detection_loss+=no_obj_gain*torch.nn.functional.mse_loss(target,pred,reduction="sum")

        iou_loss += batch_iou_loss
        classification_loss += batch_classification_loss
        obj_detection_loss += batch_obj_detection_loss
            

    return (coordinates_gain/n_all_annotations)*iou_loss, obj_detection_loss/n_all_annotations, classification_loss/n_all_annotations


    

if __name__=="__main__":
    # obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 1, True)
    obj_detect = create_potato_model(2,1)
    obj_detect.eval()
    
    input = torch.ones((1,3,512,512))
    torch.onnx.export(obj_detect, input, "obj_detect.onnx", input_names=["features"], output_names=["output"])
    print(obj_detect)





