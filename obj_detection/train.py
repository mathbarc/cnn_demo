from tqdm import tqdm

import torch
import torch.utils.data
import torch.utils.data.sampler
import torchvision

import torchmetrics.detection
import torchvision.transforms.v2

import mlflow

import model
import data_loader


def calculate_metrics(
    cnn: torch.nn.Module,
    dataset: data_loader.YoloDatasetLoader,
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


def save_model(cnn: torch.nn.Module, name: str, type: str, device):
    mlflow.pytorch.log_model(cnn, f"{name}/{type}")
    # input_sample = torch.ones((1, 3, 512, 512)).to(device)
    # model_file_name = f"{name}_{type}.onnx"
    # torch.onnx.export(
    #     cnn,
    #     input_sample,
    #     model_file_name,
    #     input_names=["features"],
    #     output_names=["output"],
    # )
    # mlflow.log_artifact(model_file_name, f"onnx/{model_file_name}")


def train_object_detector(
    cnn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    validation_dataset: data_loader.YoloDatasetLoader,
    total_step: int,
    lr: float = 0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    coordinates_loss_gain: float = 1.,
    classification_loss_gain: float = 1.,
    obj_loss_gain: float = 5.,
    no_obj_loss_gain: float = 1.,
    batches_per_step:int = 1,
):
    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr, 0.9, weight_decay=0.0005)
    # optimizer = torch.optim.ASGD(cnn.parameters(), lr)
    
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2000, 4000, 8000],0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_step, 1e-8)

    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.1),
                                                torchvision.transforms.v2.RandomEqualize(0.2),
                                                torchvision.transforms.v2.RandomResize(380,642)
                                                ])
    
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
    }

    mlflow.log_params(training_params)
    best_map = 0

    cnn.train()

    n_elements = (batch_size*batches_per_step)

    for i_step in tqdm(range(total_step)):
        optimizer.zero_grad()

        batch_position_loss = 0
        batch_scale_loss = 0
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

            # images = transform(images.to(device))
            images = images.to(device)
            annotations = annotations.to(device)

            detections = cnn(images)

            (
                position_loss, scale_loss, classification_loss, obj_detection_loss
            ) = model.calc_obj_detection_loss(
                detections,
                annotations,
                coordinates_gain=coordinates_loss_gain,
                classification_gain=classification_loss_gain,
                obj_gain=obj_loss_gain,
                no_obj_gain=no_obj_loss_gain,
                parallel=True
            )

            batch_total_loss = position_loss + scale_loss + obj_detection_loss + classification_loss
            total_loss += batch_total_loss

            batch_position_loss += position_loss.item()
            batch_scale_loss += scale_loss.item()
            batch_obj_detection_loss += obj_detection_loss.item()
            batch_classification_loss += classification_loss.item()

        total_loss.backward()
        optimizer.step()

        metrics = {
            "total_loss": total_loss.item(),
            "position_loss": batch_position_loss,
            "scale_loss": batch_scale_loss,
            "class_loss": batch_classification_loss,
            "object_presence_loss": batch_obj_detection_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        if i_step % 1000 == 999:

            performance_metrics = calculate_metrics(
                cnn, dataloader.dataset, device
            )
            metrics["train_map"] = performance_metrics

            performance_metrics = calculate_metrics(
                cnn, validation_dataset, device
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
    

    batch_size = 8
    cnn = model.create_yolo_v2_model(2,[[1.5686,1.5720], [4.1971,4.4082], [7.4817,7.1439], [11.1631,8.4289], [12.9989,12.5632]])

    dataset_train = data_loader.YoloDatasetLoader(
        "obj_detection/dataset", batch_size
    )
    dataset_valid = data_loader.YoloDatasetLoader(
        "obj_detection/dataset",
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
        cnn, dataloader, dataset_valid, 10000, 1e-3, batches_per_step=8, no_obj_loss_gain=0.5
    )
