import os

import numpy
import cv2

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torchvision.io import read_image

class CocoDataset(Dataset):
    def __init__(self, image_folder="/tmp/coco/", annotations_file=None, download_before=False):
        
        self._coco = COCO(annotations_file)
        self.img_ids = list(self._coco.imgs.keys())
        self.image_folder = image_folder

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.set_size((640,640),(19,19))

        if download_before:
            self._coco.download(self.image_folder, self.img_ids)
    
    def set_size(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    
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

        boxes = [((ann["bbox"][0] + (ann["bbox"][2]/2))/coco_img["width"], 
                  (ann["bbox"][1] + (ann["bbox"][3]/2))/coco_img["height"], 
                  (ann["bbox"][2]/coco_img["width"]), 
                  (ann["bbox"][3]/coco_img["height"])) 
                  for ann in coco_ann]
        labels = [ann["category_id"] for ann in coco_ann]

        img = resize(img,self.input_size)

        return img, {"boxes": boxes, "labels": labels}
    
    def __len__(self):
        return len(self.img_ids)

if __name__=="__main__":
    dataset = CocoDataset("/data/hd1/Dataset/Coco/images","/data/hd1/Dataset/Coco/annotations/instances_train2017.json")
    print(len(dataset))

    img, ann = dataset[0]
    