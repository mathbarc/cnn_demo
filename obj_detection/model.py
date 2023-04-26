import torch
import torchvision



def create_cnn_obj_detector_with_efficientnet_backbone(n_classes, n_masks, pretrained=True, train_backbone=True):
    efficient_net = torchvision.models.efficientnet_v2_s(pretrained=pretrained)
    for param in efficient_net.parameters():
        param.requires_grad_(train_backbone)

    backbone = list(efficient_net.children())[:-2]
    cnn = torch.nn.Sequential(*backbone)

    n_last_layer = efficient_net.features[7][0].out_channels
    cnn.add_module("output", torch.nn.Conv2d(n_last_layer, (n_classes+4)*n_masks, 1,1))
    return cnn
        
def get_transforms_for_obj_detector_with_efficientnet_backbone():
    return torchvision.transforms.Compose([ 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(512),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])



if __name__=="__main__":
    obj_detect = create_cnn_obj_detector_with_efficientnet_backbone(2, 3, True)
    input = torch.ones((1,3,512,512))
    torch.onnx.export(obj_detect, input, "obj_detect.onnx", input_names=["features"], output_names=["output"])
    print(obj_detect)





