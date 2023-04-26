import torch
import cv2
import os
from tqdm import tqdm

from enum import Enum

import logging

class DataloaderMode(Enum):
    TRAIN=0
    VALID=1
    TEST=2

class YoloDatasetLoader(torch.utils.data.Dataset):

    def __init__(self, dataset_path, transform, mode = DataloaderMode.TRAIN, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        if mode == DataloaderMode.TRAIN:
            dataset_file = "train.txt"
        elif mode == DataloaderMode.VALID:
            dataset_file = "valid.txt"        
        elif mode == DataloaderMode.TEST:
            dataset_file = "test.txt"

        with open(os.path.join(dataset_path,"classes.txt"), 'r') as file:
            self.labels = file.readlines()
            i = 0
            while i<len(self.labels):
                self.labels[i] = self.labels[i].rstrip()
                i+=1

        with open(os.path.join(dataset_path,dataset_file), 'r') as file:
            file_list = file.readlines()

        print(f"Checking images on dataset folder: {dataset_path}")

        i = 0
        while(i<len(file_list)):
            file_list[i] = file_list[i].rstrip()
            image_name = file_list[i]
            image_path = os.path.join(dataset_path, image_name)
            if not os.path.exists(image_path):
                logging.error(f"{image_name} not found!")
                del file_list[i]
                i-=1
            annotation_path = os.path.splitext(image_path)[0]+".txt"
            if not os.path.exists(annotation_path):
                logging.error(f"{image_name} does not have annotation!")
                del file_list[i]
                i-=1
            i+=1

        self.dataset_path = dataset_path
        self.file_list = file_list
        self.transform = transform
        self.device = device

    def __getitem__(self,index):
        image_name = self.file_list[index]
        image_path = os.path.join(self.dataset_path, image_name)
        annotation_path = os.path.splitext(image_path)[0]+".txt"

        img = self.transform(cv2.imread(image_path))
        annotations = YoloDatasetLoader._get_annotations_from_file(annotation_path, len(self.labels))

        return img, annotations
    
    def __len__(self):
        return len(self.file_list)
        

    def _get_annotations_from_file(annotation_path:str, n_classes:int):

        with open(annotation_path,"r") as annotation_file:
            annotation_strs = annotation_file.readlines()
        
        annotations = torch.zeros((len(annotation_strs), 4+n_classes))
        for index, annotation_str in enumerate(annotation_strs):
            annotation_values = annotation_str.rstrip().split(" ")
            label = int(annotation_values[0])
            x = float(annotation_values[1])
            y = float(annotation_values[2])
            deltaX = float(annotation_values[3])/2
            deltaY = float(annotation_values[4])/2

            annotations[index][0] = x - deltaX
            annotations[index][1] = y - deltaY
            annotations[index][2] = x + deltaX            
            annotations[index][3] = y + deltaY
            annotations[index][3+label] = 1
        return annotations


            
if __name__=="__main__":

    import model


    dataset = YoloDatasetLoader("obj_detection/dataset", model.get_transforms_for_obj_detector_with_efficientnet_backbone())


    print(dataset[0])
    print(len(dataset))
        


