import torch
import torchvision
import os
import numpy

from enum import Enum

import logging


class DataloaderMode(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class YoloDatasetLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 16,
        mode=DataloaderMode.TRAIN,
        transform: torchvision.transforms.Compose = None
    ):
        if mode == DataloaderMode.TRAIN:
            dataset_file = "train.txt"
        elif mode == DataloaderMode.VALID:
            dataset_file = "valid.txt"
        elif mode == DataloaderMode.TEST:
            dataset_file = "test.txt"

        self.batch_size = batch_size
        self.pixel_value_scale = (1.0/255.)

        with open(os.path.join(dataset_path, "classes.txt"), 'r') as file:
            self.labels = file.readlines()
            i = 0
            while i < len(self.labels):
                labels = self.labels[i].rstrip()
                self.labels[i] = labels
                i += 1

        with open(os.path.join(dataset_path, dataset_file), 'r') as file:
            file_list = file.readlines()

        print(f"Checking images on dataset folder: {dataset_path}")

        self.n_annotations = []

        i = 0
        while i < len(file_list):
            file_list[i] = file_list[i].rstrip()
            image_name = file_list[i]
            image_path = os.path.join(dataset_path, image_name)
            if not os.path.exists(image_path):
                logging.error(f"{image_name} not found!")
                del file_list[i]
                i -= 1
            annotation_path = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.exists(annotation_path):
                logging.error(f"{image_name} does not have annotation!")
                del file_list[i]
                i -= 1
            else:
                with open(annotation_path, "r") as ann_file:
                    lines = ann_file.readlines()
                    self.n_annotations.append(len(lines))
            i += 1

        self.n_annotations = numpy.unique(self.n_annotations)
        self.dataset_path = dataset_path
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.file_list[index]
        image_path = os.path.join(self.dataset_path, image_name)
        annotation_path = os.path.splitext(image_path)[0] + ".txt"

        img = torchvision.io.read_image(image_path) * self.pixel_value_scale

        if self.transform is not None:
            img = self.transform(img)

        annotations = YoloDatasetLoader._get_annotations_from_file(
            annotation_path, len(self.labels)
        )

        return img, annotations

    def __len__(self):
        return len(self.file_list)

    def get_train_indices(self):
        sel_length = numpy.random.choice(self.n_annotations)
        all_indices = numpy.where(
            [
                self.n_annotations[i] == sel_length
                for i in numpy.arange(len(self.n_annotations))
            ]
        )[0]
        indices = list(numpy.random.choice(all_indices, size=self.batch_size))
        return indices

    def _get_annotations_from_file(annotation_path: str, n_classes: int):
        with open(annotation_path, "r") as annotation_file:
            annotation_strs = annotation_file.readlines()

        annotations = torch.zeros((len(annotation_strs), 4 + n_classes))
        for index, annotation_str in enumerate(annotation_strs):
            annotation_values = annotation_str.rstrip().split(" ")
            label = int(annotation_values[0])
            cx = float(annotation_values[1])
            cy = float(annotation_values[2])
            w = float(annotation_values[3])
            h = float(annotation_values[4])

            annotations[index][0] = cx
            annotations[index][1] = cy
            annotations[index][2] = w
            annotations[index][3] = h
            annotations[index][4 + label] = 1
        return annotations


if __name__ == "__main__":
    import model
    import cv2

    # dataset_path = "/data/ssd1/Dataset/Syngenta/research-python-dataset-registry/pest_count/dataset_modelo_v2"
    dataset_path = "obj_detection/dataset"

    dataset = YoloDatasetLoader(
        dataset_path, model.get_transforms_for_obj_detector()
    )
    indexes = dataset.get_train_indices()

    img, anns = dataset[99]

    img = (img.permute(1, 2, 0).numpy() * 255).astype(numpy.uint8).copy()

    print(img.shape)

    for ann in anns:
        point = ann[0:4].numpy()
        point[0] = point[0] * img.shape[1]
        point[1] = point[1] * img.shape[0]
        point[2] = point[2] * img.shape[1]
        point[3] = point[3] * img.shape[0]
        point = point.astype(int)
        cv2.rectangle(
            img,
            (point[0], point[1], point[2] - point[0], point[3] - point[1]),
            (0, 255, 0),
            3,
        )

    cv2.imshow("img", img)
    cv2.waitKey()

    print(dataset.labels)
    print(len(dataset))
    print(indexes[0:10])
    print(dataset[indexes[0]])
