import math
import warnings
import torch
import torch.utils.data
import torch.utils.data.sampler

import numpy

import torchvision

from tqdm import tqdm

import model
import data_loader

import mlflow







def train_object_detector(cnn:torch.nn.Module, dataloader:torch.utils.data.DataLoader, total_step:int, lr:float=0.001, batchs_per_step=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), coordinates_loss_gain:float=5., no_obj_loss_gain:float=.5):

    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(),lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, total_step/10)
    
    mlflow.set_tracking_uri("http://mlflow.cluster.local")
    experiment = mlflow.get_experiment_by_name("Object Detection")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Object Detection")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    training_params = {
        "opt":str(optimizer),
        "batch_size":dataloader.batch_sampler.batch_size,
        "batches_per_step":batchs_per_step,
        "lr": lr,
        "n_steps":total_step,
        "coordinates_loss_gain":coordinates_loss_gain,
        "no_obj_loss_gain":no_obj_loss_gain
    }

    mlflow.log_params(training_params)

    cnn.train()
        
    for i_step in tqdm(range(total_step)):
        cnn.zero_grad()
        acc_loss = 0
        acc_iou_loss = 0
        acc_class_loss = 0
        acc_obj_detection = 0
        for step_batch_i in range(batchs_per_step):

            indices = dataloader.dataset.get_train_indices()
            while(len(indices)<dataloader.batch_sampler.batch_size):
                indices = dataloader.dataset.get_train_indices()
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            dataloader.batch_sampler.sampler = new_sampler

            images, annotations = next(iter(dataloader))

            images = images.to(device)
            annotations = annotations.to(device)

            outputs = cnn(images)

            iou_loss, obj_detection_loss, classification_loss = model.calc_obj_detection_loss(outputs, annotations, 1, coordinates_gain=coordinates_loss_gain, no_obj_gain=no_obj_loss_gain)
            
            total_loss = iou_loss + obj_detection_loss + classification_loss
            acc_loss += total_loss.item()
            (total_loss*(1./batchs_per_step)).backward()
            
            acc_iou_loss += iou_loss.item()
            acc_class_loss += classification_loss.item()
            acc_obj_detection += obj_detection_loss.item()

        mlflow.log_metrics({"total_loss":acc_loss, "iou_loss":acc_iou_loss, "class_loss":acc_class_loss, "object_presence_loss":acc_obj_detection}, i_step)
        
        optimizer.step()
        scheduler.step()

        if(i_step%1000 == 999):
            mlflow.pytorch.log_model(cnn, f"{i_step+1}/object_detection")
            input_sample = torch.ones((1,3,512,512)).to(device)
            torch.onnx.export(cnn, input_sample, f"object_detection_{i_step+1}.onnx", input_names=["features"], output_names=["output"])



if __name__=="__main__":


    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 
    cnn = model.create_cnn_obj_detector_with_efficientnet_backbone(2, pretrained=True)

    transforms = model.get_transforms_for_obj_detector_with_efficientnet_backbone()
    dataset = data_loader.YoloDatasetLoader("obj_detection/dataset", transforms, 8)

    indices = dataset.get_train_indices()

    initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_sampler=torch.utils.data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))

    train_object_detector(cnn, dataloader, 10000, 1e-5,4)
    
    