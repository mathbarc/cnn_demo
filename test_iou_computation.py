from matplotlib.pyplot import box
from obj_detection.model import (
    create_annotations_batch,
    generate_target_from_anotation,
    box_iou,
)
import torch


def test_iou_computation():

    annotations = {
        "boxes": torch.Tensor([[0.1, 0.1, 0.1, 0.1]]),
        "labels": torch.Tensor([[1, 0]]),
    }

    grid_size = [1, 10, 10, 7]

    det = torch.zeros(grid_size, dtype=torch.float32)

    det[0, 0, 0, 0:4] = torch.Tensor([[0.09, 0.09, 0.09, 0.09]])

    out = generate_target_from_anotation(
        annotations, grid_size, torch.Tensor([[0.1, 0.1]])
    )

    ious = box_iou(det, out)
    assert ious.sum().item() == 0.0


if __name__ == "__main__":

    test_iou_computation()
