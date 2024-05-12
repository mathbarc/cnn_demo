import torch
import torch.utils.data
import torchvision
import torchmetrics.detection
from tqdm import tqdm

import data_loader
import model

import mlflow

def calculate_metrics(
    cnn: torch.nn.Module,
    dataset: data_loader.CocoDataset,
    device: torch.device,
):
    cnn.eval()

    results = []
    targets = []

    metrics_calculator = torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh",class_metrics=True).to(device=device)

    with torch.no_grad():
        for img, ann in tqdm(dataset):
            img = torch.unsqueeze(img, 0).to(device)
            detections = cnn(img).squeeze().to("cpu")
            
            boxes = detections[:,0:4]
            objectiviness = detections[:,4]
            classes = detections[:,5:]

            result = {
                "boxes": boxes,
                "labels": torch.argmax(
                    classes, 1
                ).int(),
                "scores": objectiviness,
            }

            # best_boxes = torchvision.ops.nms(result["boxes"], result["scores"], 0.4)

            # result["boxes"] = torch.index_select(result["boxes"], 0, best_boxes)
            # result["labels"] = torch.index_select(result["labels"], 0, best_boxes)
            # result["scores"] = torch.index_select(result["scores"], 0, best_boxes)
            results.append(result)

            target = {"boxes": ann[:, :4], "labels": torch.argmax(ann[:, 4:], 1).int()}
            targets.append(target)

        metrics_calculator.update(results, targets)
        metrics = metrics_calculator.compute()
    cnn.train()
    return metrics["map"].item()

def train(dataloader : data_loader.ObjDetectionDataLoader, 
          validation_dataset : data_loader.CocoDataset,
        cnn : model.YoloV2, 
        lr : float = 0.001,
        epochs : int = 1000, 
        gradient_clip: float = 0.5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        coordinates_loss_gain: float = 1.,
        classification_loss_gain: float = 1.,
        obj_loss_gain: float = 1.,
        no_obj_loss_gain: float = 1.,
        ):

    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-8)

    best_map = 0

    # mlflow.set_tracking_uri("http://mlflow.cluster.local")
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
        "no_obj_loss_gain": no_obj_loss_gain,
        "gradient_clip": gradient_clip
    }
    
    mlflow.log_params(training_params)

    

    epoch_total_loss = 0
    epoch_position_loss = 0
    epoch_scale_loss = 0
    epoch_obj_detection_loss = 0
    epoch_classification_loss = 0

    batch_counter = 0
    for i in range(epochs):

        cnn.train()

        for imgs, anns in tqdm(dataloader):
            
            imgs = imgs.to(device)

            output = cnn(imgs)

            (
                position_loss, obj_detection_loss, classification_loss
            ) = model.calc_obj_detection_loss(
                output,
                anns,
                coordinates_gain=coordinates_loss_gain,
                classification_gain=classification_loss_gain,
                obj_gain=obj_loss_gain,
                no_obj_gain=no_obj_loss_gain,
                parallel=False
            )

            total_loss = position_loss + obj_detection_loss + classification_loss

            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_obj_detection_loss += obj_detection_loss.item()
            epoch_classification_loss += classification_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), gradient_clip)
            optimizer.step()

            batch_counter+=1
            metrics = {
                "total_loss": total_loss.item(),
                "position_loss": position_loss.item(),
                "class_loss": classification_loss.item(),
                "object_presence_loss": obj_detection_loss.item()
            }
            
            mlflow.log_metrics(metrics, batch_counter)

        scheduler.step()
        cnn.eval()
        
        metrics = {
                "epoch_total_loss": epoch_total_loss,
                "epoch_position_loss": epoch_position_loss,
                "epoch_class_loss": epoch_classification_loss,
                "epoch_object_presence_loss": epoch_obj_detection_loss,
                "lr": scheduler.get_last_lr()[0],
            }

        performance_metrics = calculate_metrics(
            cnn, validation_dataset, device
        )
        metrics["epoch_valid_map"] = performance_metrics
        mlflow.log_metrics(metrics, batch_counter)

        if best_map < performance_metrics:
            best_map = performance_metrics
            cnn.save_model("obj_detection_best")
        
        cnn.save_model("obj_detection_last")


        ...




if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = data_loader.CocoDataset("/data/hd1/Dataset/Coco/train2017","/data/hd1/Dataset/Coco/annotations/instances_train2017.json")
    validation_dataset = data_loader.CocoDataset("/data/hd1/Dataset/Coco/val2017","/data/hd1/Dataset/Coco/annotations/instances_val2017.json")
    
    # dataset = data_loader.CocoDataset("/data/hd1/Dataset/leafs/images","/data/hd1/Dataset/leafs/annotations/instances_Train.json")
    # validation_dataset = data_loader.CocoDataset("/data/hd1/Dataset/leafs/images","/data/hd1/Dataset/leafs/annotations/instances_Test.json")
    
    dataloader = data_loader.ObjDetectionDataLoader(dataset, 32, 368, 512)

    cnn = model.YoloV2(3, dataset.get_categories_count(), [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]])

    train(dataloader, validation_dataset, cnn, 1e-3)



    ...
