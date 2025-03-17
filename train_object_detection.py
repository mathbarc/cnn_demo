import torch
from obj_detection import train, model, data_loader
from obj_detection.lr_functions import (
    YoloObjDetectionRampUpLR,
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = data_loader.CocoDataset(
        "/data/ssd1/Datasets/Coco/train2017",
        "/data/ssd1/Datasets/Coco/annotations/instances_train2017.json",
    )
    validation_dataset = data_loader.CocoDataset(
        "/data/ssd1/Datasets/Coco/val2017",
        "/data/ssd1/Datasets/Coco/annotations/instances_val2017.json",
    )

    dataloader = data_loader.ObjDetectionDataLoader(dataset, 32, 368, 512)

    cnn = model.YoloV2(3, dataset.get_categories_count(), dataset.compute_anchors(4))

    lr = 1e-3
    lr_rampup_period = 1000
    epochs = 8
    obj_loss_gain = 1.0
    no_obj_loss_gain = 0.5
    classification_loss_gain = 1
    coordinates_loss_gain = 1

    optimizer = torch.optim.SGD(cnn.parameters(), lr, momentum=9e-1, weight_decay=5e-4)
    scheduler = YoloObjDetectionRampUpLR(
        optimizer,
        {
            1000: 1e-3,
            len(dataloader): 1e-4,
            2 * len(dataloader): 1e-5,
            4 * len(dataloader): 1e-6,
        },
        1e-2,
        lr_rampup_period,
        1e-8,
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
        2.0,
    )
