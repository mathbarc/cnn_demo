import os

from pycocotools.coco import COCO

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.io import read_image

import random
import time

class CocoDataset(Dataset):
    def __init__(self, image_folder="/tmp/coco/", annotations_file=None, download_before=False):
        
        self._coco = COCO(annotations_file)
        self.img_ids = list(self._coco.imgs.keys())
        self.image_folder = image_folder

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        if download_before:
            self._coco.download(self.image_folder, self.img_ids)
    
    
    def select_images_with_n_anotations(self, n):
        self.img_ids = [img for img in self._coco.imgs if len(self._coco.imgToAnns[img])==n]

    def get_categories_count(self):
        return len(self._coco.cats)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        coco_img = self._coco.imgs[img_id]
        img_path = os.path.join(self.image_folder, coco_img["file_name"])
        if not os.path.exists(img_path):
            self._coco.download(self.image_folder, [img_id])
        
        img = read_image(img_path)
        
        list_coco_ann = self._coco.getAnnIds(img_id)
        coco_ann = self._coco.loadAnns(list_coco_ann)

        boxes = [[(ann["bbox"][0] + (ann["bbox"][2]/2))/coco_img["width"], 
                  (ann["bbox"][1] + (ann["bbox"][3]/2))/coco_img["height"], 
                  (ann["bbox"][2]/coco_img["width"]), 
                  (ann["bbox"][3]/coco_img["height"])] 
                  for ann in coco_ann]
        boxesTensor = torch.FloatTensor(boxes)
        labels = [ann["category_id"] for ann in coco_ann]

        return img, {"boxes": boxesTensor, "labels": labels}
    
    def __len__(self):
        return len(self.img_ids)


class ObjDetectionDataLoader:
    def __init__(self, objDetectionDataset, batch_size, min_input_size, max_input_size):

        self.objDetectionDataset = objDetectionDataset
        self.batch_size = batch_size
        self.min_input_size = min_input_size
        self.max_input_size = max_input_size
    
    def __iter__(self):

        self.order = [i for i in range(len(self.objDetectionDataset))]
        random.shuffle(self.order)
        self.iteration_size = random.randint(self.min_input_size, self.max_input_size)
        self.iteration_transform = torchvision.transforms.Resize((self.iteration_size,self.iteration_size))


        self.position = 0

        return self
    
    @staticmethod
    def _grayscale_to_rgb(tensor:torch.Tensor) -> torch.Tensor:
        
        if tensor.shape[0] != 3:
            outputTensor = torch.cat((tensor,tensor,tensor))
        else:
            outputTensor = tensor
        
        outputTensor = outputTensor.unsqueeze(dim=0)

        return outputTensor


    def __next__(self):
        with torch.no_grad():
            start = self.position
            datasetSize = len(self.objDetectionDataset)

            if start >= datasetSize:
                raise StopIteration
            
            end = min(self.position+self.batch_size, datasetSize)

            annotations = []

            inputData, ann = self.objDetectionDataset[self.order[start]]
            inputData = self.iteration_transform(inputData)

            inputData = ObjDetectionDataLoader._grayscale_to_rgb(inputData)

            annotations.append(ann)
            

            for i in range(start+1, end):
                tmpIn, tmpAnn = self.objDetectionDataset[self.order[i]]
                tmpIn = self.iteration_transform(tmpIn)

                tmpIn = ObjDetectionDataLoader._grayscale_to_rgb(tmpIn)
                
                inputData = torch.cat((inputData, tmpIn))
                annotations.append(tmpAnn)
            
            self.position = end

            inputData = inputData.float()*(1./255.)

        return inputData, annotations

        

if __name__=="__main__":
    dataset = CocoDataset("/data/hd1/Dataset/Coco/images","/data/hd1/Dataset/Coco/annotations/instances_train2017.json")
    dataloader = ObjDetectionDataLoader(dataset, 8, 368, 512)

    i = 0

    start = time.time()
    for imgs, ann in dataloader:
        if i == 10:
            break
        i+=1
    end = time.time()

    print(end-start)
        