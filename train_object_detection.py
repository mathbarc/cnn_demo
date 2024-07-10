import torch
from obj_detection import train, model, data_loader

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

    train.train(dataloader, validation_dataset, cnn, 1e-3,100, gradient_clip=None, lr_ramp_down=1000, obj_loss_gain=1., no_obj_loss_gain=.05, classification_loss_gain=1., coordinates_loss_gain=1.)



    ...