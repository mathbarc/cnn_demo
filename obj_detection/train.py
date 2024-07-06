import torch
import torch.utils.data
import torchvision
import torchmetrics.detection

import torchvision.transforms.functional
from tqdm import tqdm
import math

import data_loader
import model

import mlflow


class ObjDetectionRampUpLR:
    def __init__(self, optimizer:torch.optim.Optimizer, lr:float, rampup_period:int, power:int = 4):
        self._optimizer = optimizer
        
        self._lr_base = lr
        self._current_step = 0
        
        self._rampup_period = rampup_period
        self._power = power
    
    def step(self):
        
        lr = self.get_last_lr()
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self._current_step+=1
            
    def get_last_lr(self):
        
        if self._current_step >= self._rampup_period:
            lr = self._lr_base
            
        elif self._current_step < self._rampup_period:
            lr = self._lr_base * pow((self._current_step / self._rampup_period), self._power)
            
        return lr


class ObjDetectionLR:
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_overshoot:float, overshoot_period:int):
        self._optimizer = optimizer
        
        self._lr_base = lr_base
        self._lr_overshoot = lr_overshoot
        
        self._current_step = 0
        self._overshoot_period = overshoot_period
        
        self._overshoot_amplitude = self._lr_overshoot - self._lr_base
    
    def step(self):
        
        lr = self.get_last_lr()
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self._current_step+=1
            
    def get_last_lr(self):
        
        if self._current_step >= self._overshoot_period:
            lr = self._lr_base
            
        elif self._current_step < self._overshoot_period:
            lr = self._lr_base + self._overshoot_amplitude * math.sin((math.pi)*(self._current_step/(self._overshoot_period)))
            
        return lr
    
class ObjDetectionDecayLR:
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_overshoot:float, lr_final:float, n_steps:int, overshoot_period:int, power:int=4):
        self._optimizer = optimizer
        
        self._lr_base = lr_base
        self._lr_overshoot = lr_overshoot
        
        self._n_steps = n_steps
        self._current_step = 0
        self._rampup_period = overshoot_period
        self._decay_period = n_steps-overshoot_period
        
        self._overshoot_amplitude = self._lr_overshoot - self._lr_base
        self._decay_amplitude = lr_final - self._lr_base
        self._lr_final = lr_final
        self._power = power
    
    def step(self):
        
        lr = self.get_last_lr()
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self._current_step+=1
            
    def get_last_lr(self):
        
        if self._current_step >= self._rampup_period:
            # lr = self._lr_base + self._decay_amplitude * math.sin((math.pi/2)*((self._current_step-self._rampup_period)/self._decay_period))
            lr = self._lr_final + self._decay_amplitude * pow(((self._decay_period-(self._current_step-self._rampup_period)) / self._decay_period), self._power)
            
        elif self._current_step < self._rampup_period:
            lr = self._lr_base * pow((self._current_step / self._rampup_period), self._power)
        
        
        
        return lr

class ObjDetectionCosineAnnealingLR:
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_overshoot:float, lr_final:float, overshoot_period:int, cosine_period:int, cosine_period_inc:float):
        self._optimizer = optimizer
        
        self._lr_base = lr_base
        self._lr_overshoot = lr_overshoot

        self._current_step = 0
        self._overshoot_period = overshoot_period
    
        self._cosine_period = cosine_period
        self._cosine_period_inc = cosine_period_inc
        
        self._overshoot_amplitude = self._lr_overshoot - self._lr_base
        self._decay_amplitude = lr_final - self._lr_base
    
    def step(self):
        
        lr = self.get_last_lr()
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self._current_step+=1
            
    def get_last_lr(self):
        
        if self._current_step >= self._overshoot_period:
            pos_in_interval = (self._current_step-self._overshoot_period)%self._cosine_period
            lr = self._lr_base + self._decay_amplitude * math.sin((math.pi/2)*(pos_in_interval/self._cosine_period))
                
            
        elif self._current_step < self._overshoot_period:
            lr = self._lr_base + self._overshoot_amplitude * math.sin((math.pi)*(self._current_step/self._overshoot_period))
        
        
        
        return lr


def calculate_metrics(
    cnn: torch.nn.Module,
    dataset: data_loader.CocoDataset,
    device: torch.device,
):
    cnn.eval()

    metrics_calculator = torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh",iou_thresholds=[0.5], rec_thresholds=[0.4],class_metrics=True).to(device=device)
    
    with torch.no_grad():
        for img, ann in tqdm(dataset):
            img = torchvision.transforms.functional.resize(img,(416,416))
            img = torch.unsqueeze(img, 0).to(device)
            detections = cnn(img).squeeze().to("cpu")
            detections = detections.reshape((detections.shape[0]*detections.shape[1]*detections.shape[2],detections.shape[3]))
            
            boxes = torch.FloatTensor(size=(0,4))
            objectiviness = torch.FloatTensor(size=(0,1))
            classes = torch.IntTensor(size=(0,1))

            for i in range(detections.shape[0]):
                if detections[i,4]>0.5:
                    boxes = torch.cat((boxes, detections[i,0:4].unsqueeze(0)))
                    objectiviness = torch.cat((objectiviness, detections[i,4].view(1,1)))
                    classes = torch.cat((classes, (torch.argmax(detections[i,5:]).int()).view(1,1)))
                    

            result = {
                "boxes": boxes,
                "labels": classes[:,0],
                "scores": objectiviness[:,0],
            }

            # best_boxes = torchvision.ops.nms(result["boxes"], result["scores"], 0.5)

            # result["boxes"] = torch.index_select(result["boxes"], 0, best_boxes)
            # result["labels"] = torch.index_select(result["labels"], 0, best_boxes)
            # result["scores"] = torch.index_select(result["scores"], 0, best_boxes)
            

            target = {"boxes": ann["boxes"], "labels": torch.argmax(ann["labels"], 1).int()}
            
            
            metrics_calculator.update([result], [target])
        metrics = metrics_calculator.compute()
    cnn.train()
    return metrics["map"].item()

def train(  dataloader : data_loader.ObjDetectionDataLoader, 
            validation_dataset : data_loader.CocoDataset,
            cnn : model.YoloV2, 
            lr : float = 0.001,
            epochs : int = 1000, 
            gradient_clip: float | None = None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            coordinates_loss_gain: float = 1.,
            classification_loss_gain: float = 1.,
            obj_loss_gain: float = 5.,
            no_obj_loss_gain: float = .5,
            lr_ramp_down:int = 100
        ):

    cnn = cnn.to(device)

    # optimizer = torch.optim.SGD(cnn.parameters(), lr, momentum=9e-1, weight_decay=5e-4)
    optimizer = torch.optim.Adam(cnn.parameters(), lr, weight_decay=5e-4)
    
    # scheduler = ObjDetectionRampUpLR(optimizer, lr, lr_ramp_down)
    # scheduler = ObjDetectionLR(optimizer, lr, 0.01, lr_ramp_down)
    scheduler = ObjDetectionDecayLR(optimizer, lr, 0.01, 1e-8, (epochs*len(dataloader)), lr_ramp_down)
    # scheduler = ObjDetectionCosineAnnealingLR(optimizer, lr, 0.01, 1e-8, lr_ramp_down, 1000, 2)
    
    if dataloader.objDetectionDataset.get_categories_count()>1:
        class_loss = torch.nn.functional.binary_cross_entropy
    else:
        class_loss = torch.nn.functional.mse_loss
    

    best_map = 0

    mlflow.set_tracking_uri("http://mlflow.cluster.local/")
    experiment = mlflow.get_experiment_by_name("Object Detection")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Object Detection")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    training_params = {
        "opt": str(optimizer),
        "batch_size": dataloader.batch_size,
        "lr": lr,
        "epoch": epochs,
        "coordinates_loss_gain": coordinates_loss_gain,
        "classification_loss_gain": classification_loss_gain,
        "obj_loss_gain": obj_loss_gain,
        "no_obj_loss_gain": no_obj_loss_gain,
        "gradient_clip": gradient_clip,
        "lr_ramp_down": lr_ramp_down,
    }
    
    mlflow.log_params(training_params)

    batch_counter = 0
    for i in range(epochs):

        cnn.train()

        for imgs, anns in tqdm(dataloader, total=len(dataloader)):
            
            optimizer.zero_grad()
            
            imgs = imgs.to(device)

            output = cnn(imgs)

            (
                position_loss, obj_detection_loss, classification_loss
            ) = model.calc_obj_detection_loss(
                output,
                anns,
                class_loss,
                coordinates_gain=coordinates_loss_gain,
                classification_gain=classification_loss_gain,
                obj_gain=obj_loss_gain,
                no_obj_gain=no_obj_loss_gain,
                parallel=False
            )

            total_loss = position_loss + obj_detection_loss + classification_loss
            loss = total_loss/dataloader.batch_size
            loss.backward()
            
            if gradient_clip is not None:    
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), gradient_clip)
                
            optimizer.step()
            scheduler.step()
            
                        
            if batch_counter % 100 == 99:
                cnn.save_model("last", device=device)
                
            batch_counter+=1
            
            metrics = {
                "total_loss": total_loss.item(),
                "position_loss": position_loss.item(),
                "class_loss": classification_loss.item(),
                "object_presence_loss": obj_detection_loss.item(), 
                "lr": optimizer.param_groups[0]["lr"]  
            }
            
            mlflow.log_metrics(metrics, batch_counter)

        
        cnn.eval()
        
        
        performance_metrics = calculate_metrics(
            cnn, validation_dataset, device
        )
        
        
        metrics = {
            "valid_map": performance_metrics
        }
        mlflow.log_metrics(metrics, batch_counter)

        if best_map < performance_metrics:
            best_map = performance_metrics
            cnn.save_model("obj_detection_best",device=device)
        
        cnn.save_model(f"obj_detection_last",device=device)
        


        ...




if __name__ == "__main__":
    
    

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset = data_loader.CocoDataset("/data/hd1/Dataset/Coco/train2017","/data/hd1/Dataset/Coco/annotations/instances_train2017.json")
    # validation_dataset = data_loader.CocoDataset("/data/hd1/Dataset/Coco/val2017","/data/hd1/Dataset/Coco/annotations/instances_val2017.json")
    
    dataset = data_loader.CocoDataset("/data/hd1/Dataset/leafs/images","/data/hd1/Dataset/leafs/annotations/instances_Train.json")
    validation_dataset = data_loader.CocoDataset("/data/hd1/Dataset/leafs/images","/data/hd1/Dataset/leafs/annotations/instances_Test.json")
    
    dataloader = data_loader.ObjDetectionDataLoader(dataset, 32, 368, 512)

    cnn = model.YoloV2(3, dataset.get_categories_count(), [[0.02,0.03],[0.05,0.06],[0.09,0.14],[0.19,0.2],[0.32,0.4],[0.83,0.77]])
    # cnn = model.YoloV2(3, dataset.get_categories_count(), [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]])
    # cnn = model.YoloV2(3, dataset.get_categories_count(), [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])

    train(dataloader, validation_dataset, cnn, 1e-3,100, gradient_clip=None, lr_ramp_down=1000, obj_loss_gain=1., no_obj_loss_gain=.05, classification_loss_gain=1., coordinates_loss_gain=1.)



    ...
