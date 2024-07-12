import torch
from obj_detection import train, model, data_loader
from obj_detection.train import ObjDetectionCosineAnnealingLR, ObjDetectionExponentialDecayLR, ObjDetectionCosineDecayLR, ObjDetectionLogisticDecayLR, ObjDetectionRampUpLR

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

    cnn = model.YoloV2(3, dataset.get_categories_count(), dataset.compute_anchors(6))
    # cnn = model.YoloV2(3, dataset.get_categories_count(), [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]])
    # cnn = model.YoloV2(3, dataset.get_categories_count(), [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])


    

    lr = 1e-3
    lr_rampup_period = 1000
    epochs = 100
    obj_loss_gain=1.
    no_obj_loss_gain=.05
    classification_loss_gain=1.
    coordinates_loss_gain=1.

    # optimizer = torch.optim.SGD(cnn.parameters(), lr, momentum=9e-1, weight_decay=5e-4)
    optimizer = torch.optim.Adam(cnn.parameters(), lr, weight_decay=5e-4)
    
    # scheduler = ObjDetectionCosineDecayLR(optimizer, lr, 1e-8, (epochs*len(dataloader)), lr_rampup_period)
    # scheduler = ObjDetectionExponentialDecayLR(optimizer, lr, 1e-8, (epochs*len(dataloader)), lr_rampup_period)
    # scheduler = ObjDetectionLogisticDecayLR(optimizer, lr, 1e-8, (epochs*len(dataloader)), lr_rampup_period)
    # scheduler = ObjDetectionCosineAnnealingLR(optimizer, lr, 1e-8, lr_rampup_period, 1000, 2, 4)
    scheduler = ObjDetectionRampUpLR(optimizer, lr, lr_rampup_period)


    train.train(dataloader, validation_dataset, cnn, optimizer, scheduler, lr, epochs, obj_loss_gain, no_obj_loss_gain, coordinates_loss_gain, classification_loss_gain, lr_rampup_period)



    ...