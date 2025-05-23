import os
import random

import cv2
import numpy
import torch
import torch.utils
import torchvision
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.transforms.v2.functional import grayscale_to_rgb


def calculate_anchors(dataset, n_anchors: int):
    data = []

    for i in range(len(dataset)):
        anns, img_sz = dataset.get_annotation(i)

        boxes = [
            [
                (ann["bbox"][2] / img_sz[0]),
                (ann["bbox"][3] / img_sz[1]),
            ]
            for ann in anns
        ]

        data.extend(boxes)

    data = numpy.asarray(data).astype(numpy.float32)
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, center = cv2.kmeans(
        data, n_anchors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    anchors = center.tolist()
    anchors.sort()
    return anchors


class CocoDataset(Dataset):
    def __init__(
        self, image_folder="/tmp/coco/", annotations_file=None, download_before=False
    ):
        self._coco = COCO(annotations_file)
        self.img_ids = list(self._coco.imgs.keys())
        self.image_folder = image_folder
        self._id_label_list = list(self._coco.cats)

        self._label_dict = {}

        label_count = 0
        for id in self._coco.cats:
            cat = self._coco.cats[id]
            self._label_dict[id] = {"label": cat["name"], "id": label_count}
            label_count += 1

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        if download_before:
            self._coco.download(self.image_folder, self.img_ids)

    def get_categories_count(self):
        return len(self._label_dict)

    def compute_anchors(self, n_anchors: int):
        data = []

        for img_id in self.img_ids:
            coco_img = self._coco.imgs[img_id]
            list_coco_ann = self._coco.getAnnIds(img_id)
            coco_ann = self._coco.loadAnns(list_coco_ann)

            boxes = [
                [
                    (ann["bbox"][2] / coco_img["width"]),
                    (ann["bbox"][3] / coco_img["height"]),
                ]
                for ann in coco_ann
            ]

            data.extend(boxes)

        data = numpy.asarray(data).astype(numpy.float32)
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, center = cv2.kmeans(
            data, n_anchors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        anchors = center.tolist()
        anchors.sort()
        return anchors

    def get_annotation(self, index: int):
        img_id = self.img_ids[index]
        coco_img = self._coco.imgs[img_id]
        list_coco_ann = self._coco.getAnnIds(img_id)
        coco_ann = self._coco.loadAnns(list_coco_ann)
        return coco_ann, (coco_img["width"], coco_img["height"])

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        coco_img = self._coco.imgs[img_id]
        img_path = os.path.join(self.image_folder, coco_img["file_name"])
        if not os.path.exists(img_path):
            self._coco.download(self.image_folder, [img_id])

        img = read_image(img_path).float()

        if img.shape[0] == 1:
            img = grayscale_to_rgb(img)

        img = img * (1.0 / 255.0)

        list_coco_ann = self._coco.getAnnIds(img_id)
        coco_ann = self._coco.loadAnns(list_coco_ann)

        boxes = [
            [
                (ann["bbox"][0] + (ann["bbox"][2] / 2)) / coco_img["width"],
                (ann["bbox"][1] + (ann["bbox"][3] / 2)) / coco_img["height"],
                (ann["bbox"][2] / coco_img["width"]),
                (ann["bbox"][3] / coco_img["height"]),
            ]
            for ann in coco_ann
        ]
        with torch.no_grad():
            boxesTensor = torch.FloatTensor(boxes)
            labels = torch.LongTensor(
                [self._label_dict[ann["category_id"]]["id"] for ann in coco_ann]
            )
            labels = torch.nn.functional.one_hot(
                labels, self.get_categories_count()
            ).float()

            return img, {"boxes": boxesTensor, "labels": labels}

    def __len__(self):
        return len(self.img_ids)


class ObjDetectionDataLoader:
    def __init__(
        self,
        objDetectionDataset,
        batch_size,
        min_input_size,
        max_input_size,
        random_batch: bool = False,
    ):
        self.objDetectionDataset = objDetectionDataset
        self.batch_size = batch_size
        self.min_input_size = min_input_size
        self.max_input_size = max_input_size
        self.random_batch = random_batch

    def __iter__(self):

        if self.random_batch:
            self.order = [
                random.randint(0, len(self.objDetectionDataset) - 1)
                for i in range(len(self.objDetectionDataset))
            ]
        else:
            self.order = [i for i in range(len(self.objDetectionDataset))]
            random.shuffle(self.order)

        self.change_input_size()
        self.position = 0

        return self

    def change_input_size(self):
        iteration_size = random.randint(self.min_input_size, self.max_input_size)
        self.iteration_transform = torchvision.transforms.Resize(
            (iteration_size, iteration_size)
        )
        print(f"changing input to size: ({iteration_size}, {iteration_size})")

    @staticmethod
    def _grayscale_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != 3:
            outputTensor = torch.cat((tensor, tensor, tensor))
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

            end = min(self.position + self.batch_size, datasetSize)

            # n_batch = int(start / self.batch_size)
            # if n_batch % 100 == 0:
            #     self._change_input_size()

            annotations = []

            inputData, ann = self.objDetectionDataset[self.order[start]]
            inputData = self.iteration_transform(inputData).unsqueeze(dim=0)

            annotations.append(ann)

            for i in range(start + 1, end):
                tmpIn, tmpAnn = self.objDetectionDataset[self.order[i]]
                tmpIn = self.iteration_transform(tmpIn).unsqueeze(dim=0)

                inputData = torch.cat((inputData, tmpIn))
                annotations.append(tmpAnn)

            self.position = end

            return inputData, annotations

    def __len__(self):
        return int(len(self.objDetectionDataset) / self.batch_size)


if __name__ == "__main__":
    dataset = CocoDataset(
        "/data/ssd1/Datasets/Coco/val2017",
        "/data/ssd1/Datasets/Coco/annotations/instances_val2017.json",
    )

    anchors = dataset.compute_anchors(6)
    print(anchors)

    dataloader = ObjDetectionDataLoader(dataset, 16, 368, 512, True)

    print(
        dataset.get_categories_count(),
        ": ",
        min(dataset._id_label_list),
        max(dataset._id_label_list),
    )

    print(dataset._label_dict)

    device = torch.device("cuda")

    import cv2

    for img, ann in dataloader:
        img = img.squeeze()
        img = img[0].permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        boxes = ann[0]["boxes"]

        for box in boxes:
            x = int((box[0] - box[2] * 0.5) * img.shape[1])
            y = int((box[1] - box[3] * 0.5) * img.shape[0])
            w = int(box[2] * img.shape[1])
            h = int(box[3] * img.shape[0])

            img = cv2.rectangle(img, rec=[x, y, w, h], color=(0, 255, 0))

        cv2.imshow("img", img)
        if cv2.waitKey() == 27:
            break
    ...
