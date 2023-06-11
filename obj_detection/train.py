import math
import warnings
import torch
import torch.utils.data
import torch.utils.data.sampler
import torchvision

from typing import Tuple

import torchmetrics.detection
import torchvision.transforms.v2

from tqdm import tqdm

import model
import data_loader

import mlflow


def calculate_metrics(
    cnn: torch.nn.Module,
    dataset: data_loader.YoloDatasetLoader,
    objects_per_cell: int,
    device: torch.device,
):
    cnn.eval()

    results = []
    targets = []

    metrics_calculator = torchmetrics.detection.MeanAveragePrecision(class_metrics=True).to(device=device)

    with torch.no_grad():
        for img, ann in dataset:
            img = torch.unsqueeze(img, 0).to(device)
            detections = cnn(img)
            
            boxes, objectiviness, classes = model.apply_activation_to_objects_from_output(detections, objects_per_cell)

            ann = ann.to(device=device)

            result = {
                "boxes": torchvision.ops.box_convert(boxes.flatten(0,-2), "cxcywh", 'xyxy'),
                "labels": torch.argmax(
                    classes.flatten(0,-2), 1
                ).int(),
                "scores": objectiviness.flatten(0,-1),
            }

            best_boxes = torchvision.ops.nms(result["boxes"], result["scores"], 0.4)

            result["boxes"] = torch.index_select(result["boxes"], 0, best_boxes)
            result["labels"] = torch.index_select(result["labels"], 0, best_boxes)
            result["scores"] = torch.index_select(result["scores"], 0, best_boxes)
            results.append(result)

            target = {"boxes": ann[:, :4], "labels": torch.argmax(ann[:, 4:], 1).int()}
            targets.append(target)

        metrics_calculator.update(results, targets)
        metrics = metrics_calculator.compute()
    cnn.train()
    return metrics["map"].item()


def save_model(cnn: torch.nn.Module, name: str, type: str, device):
    mlflow.pytorch.log_model(cnn, f"{name}/{type}")
    input_sample = torch.ones((1, 3, 512, 512)).to(device)
    model_file_name = f"{name}_{type}.onnx"
    torch.onnx.export(
        cnn,
        input_sample,
        model_file_name,
        input_names=["features"],
        output_names=["output"],
    )
    mlflow.log_artifact(model_file_name, f"onnx/{model_file_name}")


def train_object_detector(
    cnn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    validation_dataset: data_loader.YoloDatasetLoader,
    total_step: int,
    lr: float = 0.001,
    n_objects_per_cell=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    coordinates_loss_gain: float = 1.,
    classification_loss_gain: float = 1.,
    obj_loss_gain: float = 1.,
    no_obj_loss_gain: float = 0.5,
    batches_per_step:int = 1,
):
    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr, 0.9, 0.005)
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2000, 4000, 8000],0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_step, 1e-8)

    transform = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(0.1,0.1,0.1,0.05),torchvision.transforms.v2.RandomResize(380,642)])
    
    mlflow.set_tracking_uri("http://mlflow.cluster.local")
    experiment = mlflow.get_experiment_by_name("Object Detection")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Object Detection")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    training_params = {
        "opt": str(optimizer),
        "batch_size": dataloader.batch_sampler.batch_size,
        "batches_per_step": batches_per_step,
        "lr": lr,
        "n_steps": total_step,
        "coordinates_loss_gain": coordinates_loss_gain,
        "no_obj_loss_gain": no_obj_loss_gain,
        "n_objects_per_cell": n_objects_per_cell,
    }

    mlflow.log_params(training_params)
    best_map = 0

    cnn.train()


    for i_step in tqdm(range(total_step)):
        optimizer.zero_grad()

        batch_iou_loss = 0 
        batch_obj_detection_loss = 0 
        batch_classification_loss = 0
        total_loss = 0

        for i_batch in range(batches_per_step):
        
            indices = dataloader.dataset.get_train_indices()
            while len(indices) < dataloader.batch_sampler.batch_size:
                indices = dataloader.dataset.get_train_indices()
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            dataloader.batch_sampler.sampler = new_sampler

            images, annotations = next(iter(dataloader))

            images = transform(images.to(device))
            annotations = annotations.to(device)

            detections = cnn(images)

            (
                iou_loss,
                obj_detection_loss,
                classification_loss,
            ) = model.calc_obj_detection_loss(
                detections,
                annotations,
                n_objects_per_cell,
                coordinates_gain=coordinates_loss_gain,
                classification_gain=classification_loss_gain,
                obj_gain=obj_loss_gain,
                no_obj_gain=no_obj_loss_gain,
            )

            batch_total_loss = iou_loss + obj_detection_loss + classification_loss
            #total_loss.backward()
            total_loss += batch_total_loss

            batch_iou_loss += iou_loss.item()
            batch_obj_detection_loss += obj_detection_loss.item()
            batch_classification_loss += classification_loss.item()

        total_loss.backward()
        optimizer.step()

        metrics = {
            "total_loss": batch_iou_loss + batch_obj_detection_loss + batch_classification_loss,
            "iou_loss": batch_iou_loss,
            "class_loss": batch_classification_loss,
            "object_presence_loss": batch_obj_detection_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        if i_step % 100 == 99:

            performance_metrics = calculate_metrics(
                cnn, dataloader.dataset, n_objects_per_cell, device
            )
            metrics["train_map"] = performance_metrics

            performance_metrics = calculate_metrics(
                cnn, validation_dataset, n_objects_per_cell, device
            )
            metrics["valid_map"] = performance_metrics

            

            if best_map < performance_metrics:
                best_map = performance_metrics
                save_model(cnn, "object_detection", "best", device)

        mlflow.log_metrics(metrics, i_step)

        scheduler.step()

        if i_step % 100 == 99:
            save_model(cnn, "object_detection", "last", device)

        if i_step % 1000 == 999:
            save_model(cnn, "object_detection", str(i_step+1), device)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    n_objects_per_cell = 5
    batch_size = 8
    # cnn = model.create_cnn_obj_detector_with_efficientnet_backbone(2, n_objects_per_cell, pretrained=True)
    cnn = model.create_yolo_v2_model(2, n_objects_per_cell)

    transforms = model.get_transforms_for_obj_detector()
    dataset_train = data_loader.YoloDatasetLoader(
        "obj_detection/dataset", transforms, batch_size
    )
    dataset_valid = data_loader.YoloDatasetLoader(
        "obj_detection/dataset",
        transforms,
        batch_size,
        mode=data_loader.DataloaderMode.VALID,
    )

    indices = dataset_train.get_train_indices()

    initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        num_workers=8,
        batch_sampler=torch.utils.data.sampler.BatchSampler(
            sampler=initial_sampler,
            batch_size=dataset_train.batch_size,
            drop_last=False,
        ),
    )

    train_object_detector(
        cnn, dataloader, dataset_valid, 10000, 1e-4, n_objects_per_cell,batches_per_step=4, no_obj_loss_gain=0
    )
