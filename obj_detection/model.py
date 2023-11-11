import torch
from typing import List
import torchvision
import numpy


class YoloOutput(torch.nn.Module):

    def __init__(self, n_input_activation_maps:int, n_classes:int, anchors:List[List[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_layer_channels = (5+n_classes)*len(anchors)

        self.n_classes = n_classes
        
        self.conv = torch.nn.Conv2d(n_input_activation_maps, output_layer_channels, (1, 1), 
                                    (1, 1), bias=False)
        
        self.anchors = torch.Tensor(anchors)
        

    def forward(self, features: torch.Tensor):
        # Grid format -> B,C,H,W

        grid = self.conv(features)
        
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
        classes = torch.softmax(grid[:,:,5:],dim=2)

        final_boxes = torch.cat((x,y,w,h,obj,classes), dim=2)

        return final_boxes
    



class YoloV2(torch.nn.Module):
    def __init__(self, n_input_channels:int, n_classes:int, anchors:List[List[int]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torchvision.ops.Conv2dNormActivation(n_input_channels, 16, 3)
        self.conv2 = torchvision.ops.Conv2dNormActivation(16, 32, (3, 3), (1, 1))
        self.conv3 = torchvision.ops.Conv2dNormActivation(32, 64, (3, 3), (1, 1))
        self.conv4 = torchvision.ops.Conv2dNormActivation(64, 128, (3, 3), (1, 1))
        self.conv5 = torchvision.ops.Conv2dNormActivation(128, 256, (3, 3), (1, 1))
        self.conv6 = torchvision.ops.Conv2dNormActivation(256, 512, (3, 3), (1, 1))
        self.conv7 = torchvision.ops.Conv2dNormActivation(512, 1024, (3, 3), (1, 1))

        self.conv8 = torchvision.ops.Conv2dNormActivation(1024, 256, (1, 1), (1, 1))
        self.conv9 = torchvision.ops.Conv2dNormActivation(256, 512, (3, 3), (1, 1))
        self.output = YoloOutput(512, n_classes, anchors)

        self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))
    
    def forward(self, image):
        x = self.conv1(image)
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

        head_result = self.conv7(x)

        x = self.conv8(head_result)
        x = self.conv9(x)
        o1 = self.output(x)


        return o1



    def save_model(self, name: str = "yolov2", input_size = (416,416), device = None):
        
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
            # dynamic_axes=dynamic_params
        )

if __name__ == "__main__":
    device = torch.device("cuda")
    model = YoloV2(3, 2, [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]]).to(device)

    input_sample = torch.ones((8,3, 416,416))*0.5
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