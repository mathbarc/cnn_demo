from matplotlib.pyplot import grid
import torch
from typing import List, Tuple, Callable
import torchvision
import concurrent.futures
import mlflow
import mlflow.pytorch
from torchvision.ops import boxes


class YoloOutput(torch.nn.Module):
    def __init__(
        self,
        n_input_activation_maps: int,
        n_classes: int,
        anchors: List[List[int]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        output_layer_channels = (5 + n_classes) * len(anchors)
        self.conv = torch.nn.Conv2d(
            n_input_activation_maps, output_layer_channels, (1, 1), (1, 1)
        )
        self.n_classes = torch.tensor(n_classes, requires_grad=False)

        anchors_data = torch.tensor(anchors, dtype=torch.float32)
        self.anchors = torch.nn.Parameter(
            anchors_data.reshape((anchors_data.size(0), anchors_data.size(1), 1, 1)),
            requires_grad=False,
        )

    def forward(self, features: torch.Tensor):
        # Grid format -> B,C,H,W

        grid = self.conv(features)
        # grid = torch.nan_to_num(grid)

        with torch.no_grad():
            grid_cell_position_y, grid_cell_position_x = torch.meshgrid(
                [torch.arange(grid.size(2)), torch.arange(grid.size(3))], indexing="ij"
            )
            grid_cell_position_y = grid_cell_position_y.to(features.device).detach()
            grid_cell_position_x = grid_cell_position_x.to(features.device).detach()
            grid_dimensions = [
                grid.size(0),
                self.anchors.size(0),
                (5 + self.n_classes),
                grid.size(2),
                grid.size(3),
            ]

        grid = grid.reshape(grid_dimensions).to(features.device)
        anchors_tiled = self.anchors.to(features.device)

        x = (
            (grid_cell_position_x + torch.sigmoid(grid[:, :, 0])) / grid.size(4)
        ).unsqueeze(2)
        y = (
            (grid_cell_position_y + torch.sigmoid(grid[:, :, 1])) / grid.size(3)
        ).unsqueeze(2)
        w = (anchors_tiled[:, 0] * torch.exp(grid[:, :, 2])).unsqueeze(2)
        h = (anchors_tiled[:, 1] * torch.exp(grid[:, :, 3])).unsqueeze(2)

        obj = torch.sigmoid(grid[:, :, 4]).unsqueeze(2)
        # classes = torch.sigmoid(grid[:, :, 5:])
        classes = torch.softmax(grid[:, :, 5:], dim=2)

        final_boxes = torch.cat((x, y, w, h, obj, classes), dim=2)

        return final_boxes


class YoloV2(torch.nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_classes: int,
        anchors: List[List[int]],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torchvision.ops.Conv2dNormActivation(
            n_input_channels, 16, (3, 3), (1, 1)
        )
        self.conv2 = torchvision.ops.Conv2dNormActivation(16, 32, (3, 3), (1, 1))
        self.conv3 = torchvision.ops.Conv2dNormActivation(32, 64, (3, 3), (1, 1))
        self.conv4 = torchvision.ops.Conv2dNormActivation(64, 128, (3, 3), (1, 1))
        self.conv5 = torchvision.ops.Conv2dNormActivation(128, 256, (3, 3), (1, 1))
        self.conv6 = torchvision.ops.Conv2dNormActivation(256, 512, (3, 3), (1, 1))
        self.conv7 = torchvision.ops.Conv2dNormActivation(512, 1024, (3, 3), (1, 1))

        self.conv8 = torchvision.ops.Conv2dNormActivation(1024, 512, (3, 3), (1, 1))
        self.output = YoloOutput(512, n_classes, anchors)

        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=1)

    def forward(self, image):
        i = image.float()
        x = self.conv1(i)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.pool1(x)

        l1 = self.conv5(x)
        x = self.pool1(l1)

        x = self.conv6(x)
        x = self.pool2(x)

        head_result = self.conv7(x)

        x = self.conv8(head_result)
        o1 = self.output(x)

        o1 = torch.permute(o1, (0, 1, 3, 4, 2))

        return o1

    def save_model(self, name: str = "yolov2", input_size=(416, 416), device=None):

        input_sample = torch.ones((1, 3, input_size[1], input_size[0]))
        if device is not None:
            input_sample = input_sample.to(device)
        model_file_name = f"{name}.onnx"
        # dynamic_params = {"features":{0:"batch_size", 2:"image_height", 3:"image_width"},
        #                   "output1":{0:"batch_size",3:"grid_y",4:"grid_x"},
        #                   "output2":{0:"batch_size",3:"grid_y",4:"grid_x"}}
        torch.onnx.export(
            self,
            input_sample,
            model_file_name,
            input_names=["features"],
            output_names=["output"],
            opset_version=11,
            # dynamic_axes=dynamic_params
        )
        try:
            mlflow.pytorch.log_model(self, name, extra_files=[__file__])
            mlflow.log_artifact(model_file_name, f"onnx/{model_file_name}")
        except:
            print("skipping model log")


def compute_iou(anchor_w, anchor_h, bbox_w, bbox_h):
    inter_w = min(anchor_w, bbox_w)
    inter_h = min(anchor_h, bbox_h)
    intersection_area = inter_w * inter_h

    anchor_area = anchor_w * anchor_h
    bbox_area = bbox_w * bbox_h

    union_area = (anchor_area + bbox_area - intersection_area).clamp(min=1e-8)

    iou = intersection_area / union_area
    return iou


def generate_target_from_anotation(annotations, grid_size, anchors: torch.Tensor):
    ann_boxes = annotations["boxes"]
    ann_labels = annotations["labels"]

    target_tensor = torch.zeros(
        (grid_size[0], grid_size[1], grid_size[2], grid_size[3])
    )

    for ann_id in range(len(ann_boxes)):
        bbox = ann_boxes[ann_id]
        label = ann_labels[ann_id]

        x_center, y_center, width, height = bbox

        x_min = max(x_center - width / 2.0, 0)
        x_max = min(x_center + width / 2.0, 1)
        y_min = max(y_center - height / 2.0, 0)
        y_max = min(y_center + height / 2.0, 1)

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        grid_x = int(x_center * grid_size[2])
        grid_y = int(y_center * grid_size[1])

        if grid_x < grid_size[2] and grid_y < grid_size[1]:
            best_iou = -1
            best_anchor = -1
            for i in range(grid_size[0]):
                anchor_w = anchors[i, 0].item()
                anchor_h = anchors[i, 1].item()
                iou = compute_iou(anchor_w, anchor_h, width, height)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i

            target_tensor[best_anchor, grid_y, grid_x, 0] = x_center
            target_tensor[best_anchor, grid_y, grid_x, 1] = y_center
            target_tensor[best_anchor, grid_y, grid_x, 2] = width
            target_tensor[best_anchor, grid_y, grid_x, 3] = height
            target_tensor[best_anchor, grid_y, grid_x, 4] = 1
            target_tensor[best_anchor, grid_y, grid_x, 5:] = label

    return target_tensor


def create_annotations_batch(
    annotations: torch.Tensor, anchors: torch.Tensor, grid_size
) -> torch.Tensor:

    target = torch.zeros(grid_size)
    with torch.no_grad():
        n_batches = grid_size[0]

        for batch_id in range(n_batches):
            target[batch_id] = generate_target_from_anotation(
                annotations[batch_id], grid_size[1:], anchors
            )
    return target


def box_iou(detection, target):

    area1 = detection[..., 2] * detection[..., 3]
    det_x_min = detection[..., 0] - (detection[..., 2] * 0.5)
    det_x_max = detection[..., 0] + (detection[..., 2] * 0.5)
    det_y_min = detection[..., 1] - (detection[..., 3] * 0.5)
    det_y_max = detection[..., 1] + (detection[..., 3] * 0.5)

    with torch.no_grad():
        area2 = target[..., 2] * target[..., 3]
        target_x_min = target[..., 0] - (target[..., 2] * 0.5)
        target_x_max = target[..., 0] + (target[..., 2] * 0.5)
        target_y_min = target[..., 1] - (target[..., 3] * 0.5)
        target_y_max = target[..., 1] + (target[..., 3] * 0.5)

    left = torch.maximum(det_x_min, target_x_min)
    top = torch.maximum(det_y_min, target_y_min)

    right = torch.minimum(det_x_max, target_x_max)
    bottom = torch.minimum(det_y_max, target_y_max)

    w = (right - left).clamp(min=0)
    h = (bottom - top).clamp(min=0)

    inter = torch.mul(w, h)

    union = area1 + area2 - inter

    return torch.div(inter, union + 1e-8)


def obj_detection_loss(
    detections: torch.Tensor,
    annotations,
    anchors,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 1.0,
    no_obj_gain: float = 0.5,
    ignore_obj_thr: float = 0.2,
    filter_iou: bool = False,
):

    det_boxes = detections[..., :4]
    det_obj = detections[..., 4:5]
    det_cls = detections[..., 5:]

    with torch.no_grad():
        target = create_annotations_batch(annotations, anchors, detections.shape)
        target = target.to(detections.device)
        target_boxes = target[..., :4]
        target_obj = target[..., 4:5]
        target_cls = target[..., 5:]

        if filter_iou:
            iou = box_iou(det_boxes, target_boxes)
            ignore_iou_mask = iou < ignore_obj_thr
            target_obj[ignore_iou_mask] = 0

    use_complete_box_iou_loss = True
    if use_complete_box_iou_loss:
        coordinates_loss = 1.0 - box_iou(det_boxes, target_boxes).unsqueeze(-1)

        # coordinates_xyxy = torchvision.ops.box_convert(det_boxes, "cxcywh", "xyxy")
        # with torch.no_grad():
        #     target_xyxy = torchvision.ops.box_convert(target_boxes, "cxcywh", "xyxy")
        # coordinates_loss = torchvision.ops.generalized_box_iou_loss(
        #     coordinates_xyxy, target_xyxy, reduction="none"
        # ).unsqueeze(-1)

    else:
        coordinates_loss = torch.nn.functional.mse_loss(
            det_boxes, target_boxes, reduction="none"
        )
        coordinates_loss = torch.sqrt(torch.sum(coordinates_loss, dim=-1, keepdim=True))

    coordinates_loss = coordinates_gain * torch.sum(target_obj * coordinates_loss)

    conf_obj_loss = torch.sum(
        target_obj
        * torch.nn.functional.mse_loss(det_obj, target_obj, reduction="none"),
    )

    conf_noobj_loss = torch.sum(
        (1 - target_obj)
        * torch.nn.functional.mse_loss(det_obj, target_obj, reduction="none"),
    )

    obj_loss = (obj_gain * conf_obj_loss) + (no_obj_gain * conf_noobj_loss)

    use_binary_cross_entropy = False

    if use_binary_cross_entropy:
        cls_err = torch.nn.functional.binary_cross_entropy(
            det_cls, target_cls, reduction="none"
        )
        cls_err = torch.sum(cls_err, 4, keepdim=True)
    else:
        det_cls = torch.permute(det_cls, (0, 4, 1, 2, 3))
        target_cls = torch.permute(target_cls, (0, 4, 1, 2, 3))
        cls_err = torch.nn.functional.cross_entropy(
            det_cls, target_cls, reduction="none"
        ).unsqueeze(-1)

        # cls_err = torch.nn.functional.mse_loss(det_cls, target_cls, reduction="none")
        # cls_err = torch.sqrt(torch.sum(cls_err, 4, keepdim=True))

    cls_loss = classification_gain * torch.sum(
        target_obj * cls_err,
    )

    return coordinates_loss, obj_loss, cls_loss


def obj_detection_loss_with_batch_mean(
    detections: torch.Tensor,
    annotations,
    anchors,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 0.5,
    no_obj_gain: float = 0.5,
):

    target = create_annotations_batch(detections, annotations, anchors)
    target = target.to(detections.device)

    det_boxes = detections[..., :4]
    det_obj = detections[..., 4:5]
    det_cls = detections[..., 5:]

    target_boxes = target[..., :4]
    target_obj = target[..., 4:5]
    target_cls = target[..., 5:]

    iou = box_iou(det_boxes, target_boxes)
    ignore_iou_mask = iou < 0.2

    target_obj[ignore_iou_mask] = 0

    coordinates_loss = coordinates_gain * torch.mean(
        torch.sum(
            target_obj
            * torch.nn.functional.mse_loss(det_boxes, target_boxes, reduction="none"),
            (1, 2, 3, 4),
        )
    )

    conf_obj_loss = torch.mean(
        torch.sum(
            target_obj
            * torch.nn.functional.mse_loss(det_obj, target_obj, reduction="none"),
            (1, 2, 3, 4),
        )
    )

    conf_noobj_loss = torch.mean(
        torch.sum(
            (1 - target_obj)
            * torch.nn.functional.mse_loss(det_obj, target_obj, reduction="none"),
            (1, 2, 3, 4),
        )
    )

    obj_loss = (obj_gain * conf_obj_loss) + (no_obj_gain * conf_noobj_loss)

    use_binary_cross_entropy = False

    if use_binary_cross_entropy:
        cls_err = torch.nn.functional.binary_cross_entropy(
            det_cls, target_cls, reduction="none"
        )
        cls_err = torch.sum(cls_err, 4, keepdim=True)
    else:
        det_cls = torch.permute(det_cls, (0, 4, 1, 2, 3))
        target_cls = torch.permute(target_cls, (0, 4, 1, 2, 3))
        cls_err = torch.nn.functional.cross_entropy(
            det_cls, target_cls, reduction="none"
        ).unsqueeze(-1)

    cls_loss = classification_gain * torch.mean(
        torch.sum(
            target_obj * cls_err,
            (1, 2, 3, 4),
        )
    )

    return coordinates_loss, obj_loss, cls_loss


def calc_batch_loss(
    detections: torch.Tensor, annotations, class_loss, obj_gain, no_obj_gain
):
    batch_position_loss = torch.zeros(
        [1], device=detections.device, requires_grad=False
    )
    batch_classification_loss = torch.zeros(
        [1], device=detections.device, requires_grad=False
    )
    batch_with_obj_detection_loss = torch.zeros(
        [1], device=detections.device, requires_grad=False
    )
    batch_without_obj_detection_loss = torch.zeros(
        [1], device=detections.device, requires_grad=False
    )

    obj_boxes = detections[:, :, :, 0:4]
    obj_boxes_xyxy = torchvision.ops.box_convert(obj_boxes, "cxcywh", "xyxy")

    zero = torch.zeros([1], device=detections.device, requires_grad=False)[0]
    one = torch.ones([1], device=detections.device, requires_grad=False)

    if annotations["boxes"].shape[0] > 0:
        ann_boxes = annotations["boxes"].to(detections.device)
        ann_classes = annotations["labels"].to(detections.device)

        ann_xyxy = torchvision.ops.box_convert(ann_boxes, "cxcywh", "xyxy").detach()
        contains_obj = {}

        for i in range(ann_xyxy.size(0)):
            ann_box = ann_xyxy[i]

            cellX = int(
                (ann_box[0].item() + ann_box[2].item()) * 0.5 * detections.size(2)
            )
            cellY = int(
                (ann_box[1].item() + ann_box[3].item()) * 0.5 * detections.size(1)
            )

            iou = torchvision.ops.box_iou(
                obj_boxes_xyxy[:, cellY, cellX, :], ann_box.view(1, -1)
            )

            p = (iou.argmax().item(), cellY, cellX)

            if p[0] < 0.2:
                continue

            if p not in contains_obj:
                contains_obj[p] = (i, iou.max())
            else:
                if contains_obj[p][1] < iou.max():
                    contains_obj[p] = (i, iou.max())

        for i in range(detections.shape[0]):
            for j in range(detections.shape[1]):
                for k in range(detections.shape[2]):
                    pos = (i, j, k)

                    if pos in contains_obj:
                        ann_id, best_iou = contains_obj[pos]

                        batch_position_loss += torchvision.ops.complete_box_iou_loss(
                            obj_boxes_xyxy[i, j, k], ann_xyxy[ann_id]
                        )
                        batch_with_obj_detection_loss += torch.nn.functional.mse_loss(
                            detections[i, j, k, 4], one
                        )
                        batch_classification_loss += class_loss(
                            detections[i, j, k, 5:], ann_classes[ann_id]
                        )

                    else:
                        batch_without_obj_detection_loss += (
                            torch.nn.functional.mse_loss(detections[i, j, k, 4], zero)
                        )

    else:
        for i in range(detections.shape[0]):
            for j in range(detections.shape[1]):
                for k in range(detections.shape[2]):
                    batch_without_obj_detection_loss += torch.nn.functional.mse_loss(
                        detections[i, j, k, 4], zero
                    )

    return (
        batch_position_loss,
        batch_classification_loss,
        (obj_gain * batch_with_obj_detection_loss)
        + (no_obj_gain * batch_without_obj_detection_loss),
    )


def calc_obj_detection_loss(
    detections: torch.Tensor,
    annotations: torch.Tensor,
    class_loss: Callable = torch.nn.functional.binary_cross_entropy,
    coordinates_gain: float = 1.0,
    classification_gain: float = 1.0,
    obj_gain: float = 5.0,
    no_obj_gain: float = 0.5,
    parallel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batches = detections.size()[0]

    detections = torch.nan_to_num(detections, 0, 0, 0)

    position_loss = 0
    classification_loss = 0
    obj_detection_loss = 0

    if parallel:
        executor = concurrent.futures.ThreadPoolExecutor(n_batches)
        batch_processing = []

        for batch_id in range(n_batches):
            batch_process = executor.submit(
                calc_batch_loss,
                detections[batch_id],
                annotations[batch_id],
                class_loss,
                obj_gain,
                no_obj_gain,
            )
            batch_processing.append(batch_process)

        for batch_process in batch_processing:
            batch_position_loss, batch_classification_loss, batch_obj_detection_loss = (
                batch_process.result()
            )

            position_loss += batch_position_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss
    else:
        for batch_id in range(n_batches):
            batch_position_loss, batch_classification_loss, batch_obj_detection_loss = (
                calc_batch_loss(
                    detections[batch_id],
                    annotations[batch_id],
                    class_loss,
                    obj_gain,
                    no_obj_gain,
                )
            )

            position_loss += batch_position_loss
            classification_loss += batch_classification_loss
            obj_detection_loss += batch_obj_detection_loss

    return (
        (coordinates_gain * position_loss) / n_batches,
        obj_detection_loss / n_batches,
        (classification_gain * classification_loss) / n_batches,
    )


if __name__ == "__main__":
    device = torch.device("cuda")
    model = YoloV2(
        3, 2, [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
    )
    import cProfile
    import data_loader
    import time

    dataset = data_loader.CocoDataset(
        "/data/ssd1/Datasets/Coco/train2017",
        "/data/ssd1/Datasets/Coco/annotations/instances_train2017.json",
    )

    dataloader = data_loader.ObjDetectionDataLoader(dataset, 64, 368, 512)

    with cProfile.Profile() as profile:
        mean = 0
        count = 0
        for input_sample, target in dataloader:
            start = time.time()

            output = model(input_sample)
            loss = calc_obj_detection_loss(output, target)

            end = time.time()

            v = end - start
            mean += v
            print(v)

            time.sleep(1)
            if count == 10:
                print(mean / 10)
                count = 0
            break

        profile.dump_stats("profile.prof")

    ...
