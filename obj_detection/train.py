import torch
import torch.utils.data
import torchvision
import torchmetrics

import data_loader
import model

import mlflow


def train(dataloader : data_loader.ObjDetectionDataLoader, 
        cnn : torch.nn.Module, 
        lr : float = 0.001,
        epochs : int = 1000, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        coordinates_loss_gain: float = 1.,
        classification_loss_gain: float = 1.,
        obj_loss_gain: float = 5.,
        no_obj_loss_gain: float = 1.,
        batches_per_step:int = 1,
        ):

    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-8)

    # mlflow.set_tracking_uri("http://mlflow.cluster.local")
    # experiment = mlflow.get_experiment_by_name("Object Detection")
    # if experiment is None:
    #     experiment_id = mlflow.create_experiment("Object Detection")
    # else:
    #     experiment_id = experiment.experiment_id
    # mlflow.start_run(experiment_id=experiment_id)

    # training_params = {
    #     "opt": str(optimizer),
    #     "batch_size": dataloader.batch_size,
    #     "lr": lr,
    #     "epoch": epochs,
    #     "coordinates_loss_gain": coordinates_loss_gain,
    #     "no_obj_loss_gain": no_obj_loss_gain,
    # }
    
    # mlflow.log_params(training_params)

    cnn.train()

    for i in range(epochs):

        for imgs, anns in dataloader:

            imgs = imgs.to(device)

            output = cnn(imgs)

            ...




if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = data_loader.CocoDataset("/data/hd1/Dataset/Coco/images","/data/hd1/Dataset/Coco/annotations/instances_train2017.json")
    dataloader = data_loader.ObjDetectionDataLoader(dataset, 8, 368, 512)

    cnn = model.YoloV2(3, 2, [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]])

    train(dataloader, cnn)



    ...
