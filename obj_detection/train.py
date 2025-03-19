import torch
import torch.utils.data
import torchvision
import torchmetrics.detection

import torchvision.transforms.functional
from tqdm import tqdm
import math

from . import data_loader
from . import model

import mlflow


def calculate_metrics(
    cnn: torch.nn.Module,
    dataset: data_loader.CocoDataset,
    device: torch.device,
):
    cnn.eval()

    metrics_calculator = torchmetrics.detection.MeanAveragePrecision(
        box_format="cxcywh",
        iou_thresholds=[0.5],
        rec_thresholds=[0.25],
        class_metrics=True,
    ).to(device=device)

    with torch.no_grad():
        for img, ann in tqdm(dataset):
            img = torchvision.transforms.functional.resize(img, (416, 416))
            img = torch.unsqueeze(img, 0).to(device)
            detections = cnn(img).squeeze().to("cpu")
            detections = detections.reshape(
                (
                    detections.shape[0] * detections.shape[1] * detections.shape[2],
                    detections.shape[3],
                )
            )

            boxes = torch.FloatTensor(size=(0, 4))
            objectiviness = torch.FloatTensor(size=(0, 1))
            classes = torch.IntTensor(size=(0, 1))

            for i in range(detections.shape[0]):
                if detections[i, 4] > 0.5:
                    boxes = torch.cat((boxes, detections[i, 0:4].unsqueeze(0)))
                    objectiviness = torch.cat(
                        (objectiviness, detections[i, 4].view(1, 1))
                    )
                    classes = torch.cat(
                        (classes, (torch.argmax(detections[i, 5:]).int()).view(1, 1))
                    )

            result = {
                "boxes": boxes,
                "labels": classes[:, 0],
                "scores": objectiviness[:, 0],
            }

            # best_boxes = torchvision.ops.nms(result["boxes"], result["scores"], 0.5)

            # result["boxes"] = torch.index_select(result["boxes"], 0, best_boxes)
            # result["labels"] = torch.index_select(result["labels"], 0, best_boxes)
            # result["scores"] = torch.index_select(result["scores"], 0, best_boxes)

            target = {
                "boxes": ann["boxes"],
                "labels": torch.argmax(ann["labels"], 1).int(),
            }

            metrics_calculator.update([result], [target])
        metrics = metrics_calculator.compute()
    cnn.train()
    return metrics["map"].item()


def train(
    dataloader: data_loader.ObjDetectionDataLoader,
    validation_dataset: data_loader.CocoDataset,
    cnn: model.YoloV2,
    optimizer,
    scheduler,
    lr: float = 0.001,
    epochs: int = 1000,
    obj_loss_gain: float = 5.0,
    no_obj_loss_gain: float = 0.5,
    coordinates_loss_gain: float = 1.0,
    classification_loss_gain: float = 1.0,
    lr_rampup_period: int = 100,
    gradient_clip: float | None = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    cnn = cnn.to(device)

    if dataloader.objDetectionDataset.get_categories_count() > 1:
        class_loss = torch.nn.functional.binary_cross_entropy
    else:
        class_loss = torch.nn.functional.mse_loss

    best_map = 0

    mlflow.set_tracking_uri("http://mlflow.solv.local/")
    experiment = mlflow.get_experiment_by_name("Object Detection")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Object Detection")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    training_params = {
        "opt": str(optimizer),
        "scheduler": str(scheduler),
        "batch_size": dataloader.batch_size,
        "lr": lr,
        "epoch": epochs,
        "coordinates_loss_gain": coordinates_loss_gain,
        "classification_loss_gain": classification_loss_gain,
        "obj_loss_gain": obj_loss_gain,
        "no_obj_loss_gain": no_obj_loss_gain,
        "gradient_clip": gradient_clip,
        "lr_rampup_period": lr_rampup_period,
        "n_anchors": cnn.output.anchors.shape[0],
    }

    mlflow.log_params(training_params)

    batch_counter = 0
    for i in range(epochs):
        cnn.train()

        for imgs, anns in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()

            imgs = imgs.to(device)

            output = cnn(imgs)

            (position_loss, obj_detection_loss, classification_loss) = (
                model.obj_detection_loss(
                    output,
                    anns,
                    cnn.output.anchors,
                    coordinates_gain=coordinates_loss_gain,
                    classification_gain=classification_loss_gain,
                    no_obj_gain=no_obj_loss_gain,
                )
            )

            total_loss = position_loss + obj_detection_loss + classification_loss
            loss = total_loss
            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            if batch_counter % 500 == 499:
                cnn.save_model("last", device=device)

            if batch_counter % 10 == 0:
                metrics = {
                    "total_loss": total_loss.item(),
                    "position_loss": position_loss.item(),
                    "class_loss": classification_loss.item(),
                    "object_presence_loss": obj_detection_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                try:
                    mlflow.log_metrics(metrics, batch_counter)
                except:
                    print("skipping metrics log")

            batch_counter += 1

        cnn.eval()

        performance_metrics = calculate_metrics(cnn, validation_dataset, device)

        metrics = {"valid_map": performance_metrics}
        try:
            mlflow.log_metrics(metrics, batch_counter)
        except:
            print("skipping validation log")

        if best_map < performance_metrics:
            best_map = performance_metrics
            cnn.save_model("obj_detection_best", device=device)

        cnn.save_model("obj_detection_last", device=device)

        ...
