import logging
import torch
from obj_detection import train, model, data_loader
from obj_detection.lr_functions import (
    YoloObjDetectionRampUpLR,
)
import os

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info("Loading Training set ")
    dataset = data_loader.CocoDataset(
        os.path.join(os.environ["DATASET_PATH"], "coco_dataset/2017/train2017"),
        os.path.join(
            os.environ["DATASET_PATH"],
            "coco_dataset/2017/annotations/instances_train2017.json",
        ),
    )
    logging.info("Loading validation set ")
    validation_dataset = data_loader.CocoDataset(
        os.path.join(os.environ["DATASET_PATH"], "coco_dataset/2017/val2017"),
        os.path.join(
            os.environ["DATASET_PATH"],
            "coco_dataset/2017/annotations/instances_val2017.json",
        ),
    )

    logging.info("Creating data loader for training set")
    dataloader = data_loader.ObjDetectionDataLoader(dataset, 64, 368, 512)

    n_anchors = 5
    logging.info("Calculating anchors ... {n_anchors}")
    anchors = data_loader.calculate_anchors(dataset, n_anchors)
    logging.info(f"Found anchors: {anchors}")

    cnn = model.YoloV2(3, dataset.get_categories_count(), anchors)

    lr = 1e-3
    peak_lr = 1e-3
    start_lr = 1e-8
    lr_rampup_period = 1000
    epochs = 100
    obj_loss_gain = 1.0
    no_obj_loss_gain = 0.5
    classification_loss_gain = 1.0
    coordinates_loss_gain = 1.0

    optimizer = torch.optim.SGD(
        cnn.parameters(), start_lr, momentum=9e-1, weight_decay=5e-4
    )
    scheduler = YoloObjDetectionRampUpLR(
        optimizer,
        {
            lr_rampup_period: lr,
            50 * len(dataloader): 1e-4,
            80 * len(dataloader): 1e-5,
        },
        peak_lr,
        lr_rampup_period,
        start_lr,
    )

    train.train(
        dataloader,
        validation_dataset,
        cnn,
        optimizer,
        scheduler,
        lr,
        epochs,
        obj_loss_gain,
        no_obj_loss_gain,
        coordinates_loss_gain,
        classification_loss_gain,
        lr_rampup_period,
    )
