import os

from pycocotools.coco import COCO

import torch
import torch.utils
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.io import read_image
from torchvision.transforms.v2.functional import grayscale_to_rgb

import random
import time

class CocoDataset(Dataset):
    def __init__(self, image_folder="/tmp/coco/", annotations_file=None, download_before=False):
        
        self._coco = COCO(annotations_file)
        self.img_ids = list(self._coco.imgs.keys())
        self.image_folder = image_folder
        self._id_label_list = list(self._coco.cats)
        
        self._label_dict = {}
        
        label_count = 0
        for id in self._coco.cats:
            cat = self._coco.cats[id]
            self._label_dict[id] = {"label":cat["name"], "id":label_count}
            label_count+=1
        
        

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        if download_before:
            self._coco.download(self.image_folder, self.img_ids)
    
    
    def select_images_with_n_anotations(self, n):
        self.img_ids = [img for img in self._coco.imgs if len(self._coco.imgToAnns[img])==n]

    def get_categories_count(self):
        return len(self._label_dict)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        coco_img = self._coco.imgs[img_id]
        img_path = os.path.join(self.image_folder, coco_img["file_name"])
        if not os.path.exists(img_path):
            self._coco.download(self.image_folder, [img_id])
        
        img = read_image(img_path).float()
        
        if(img.shape[0] == 1):
            img = grayscale_to_rgb(img)
        
        img = img*(1./255.)
        
        list_coco_ann = self._coco.getAnnIds(img_id)
        coco_ann = self._coco.loadAnns(list_coco_ann)

        boxes = [[(ann["bbox"][0] + (ann["bbox"][2]/2))/coco_img["width"], 
                  (ann["bbox"][1] + (ann["bbox"][3]/2))/coco_img["height"], 
                  (ann["bbox"][2]/coco_img["width"]), 
                  (ann["bbox"][3]/coco_img["height"])] 
                  for ann in coco_ann]
        with torch.no_grad():
            boxesTensor = torch.FloatTensor(boxes)
            labels = torch.LongTensor([self._label_dict[ann["category_id"]]["id"] for ann in coco_ann])
            labels = torch.nn.functional.one_hot(labels, self.get_categories_count()).float()

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
            inputData = self.iteration_transform(inputData).unsqueeze(dim=0)

            annotations.append(ann)
            

            for i in range(start+1, end):
                tmpIn, tmpAnn = self.objDetectionDataset[self.order[i]]
                tmpIn = self.iteration_transform(tmpIn).unsqueeze(dim=0)
                
                inputData = torch.cat((inputData, tmpIn))
                annotations.append(tmpAnn)
            
            self.position = end

            return inputData, annotations

    def __len__(self):
        return int(len(self.objDetectionDataset)/self.batch_size)

if __name__=="__main__":
    dataset = CocoDataset("/data/hd1/Dataset/Coco/val2017","/data/hd1/Dataset/Coco/annotations/instances_val2017.json")
    dataloader = ObjDetectionDataLoader(dataset, 1, 368, 512)
    
    print(dataset.get_categories_count())
    device = torch.device("cuda")
    
    import cv2
    
    for img, ann in dataloader:
        img = img.squeeze()
        img = img.permute(1,2,0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        boxes = ann[0]["boxes"]
        
        for box in boxes:
            x = int((box[0] - box[2]*0.5)*img.shape[1])
            y = int((box[1] - box[3]*0.5)*img.shape[0])
            w = int(box[2]*img.shape[1])
            h = int(box[3]*img.shape[0])
            
            img = cv2.rectangle(img, rec = [x,y,w,h], color=(0,255,0))
        
        cv2.imshow("img", img)
        if cv2.waitKey() == 27:
            break
    ...    