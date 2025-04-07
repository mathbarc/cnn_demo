from matplotlib.pyplot import box
from obj_detection.model import (
    create_annotations_batch,
    generate_target_from_anotation,
    box_iou,
)
import torch
import pytest
from typing import Dict


@pytest.mark.parametrize(
    "annotations,anchors,grid_size,expected_indexes",
    [
        (
            {
                "boxes": torch.Tensor([[0.1, 0.1, 0.1, 0.1], [0.21, 0.1, 0.3, 0.4]]),
                "labels": torch.Tensor([[1, 0], [0, 1]]),
            },
            torch.Tensor([[0.1, 0.1], [0.2, 0.3]]),
            [2, 10, 10, 7],
            [(0, 1, 1), (1, 1, 2)],
        ),
        (
            {
                "boxes": torch.Tensor([[0.6, 0.6, 0.1, 0.1], [0.1, 0.1, 0.3, 0.4]]),
                "labels": torch.Tensor([[1], [0]]),
            },
            torch.Tensor([[0.1, 0.1], [0.2, 0.3]]),
            [2, 2, 2, 6],
            [(0, 1, 1), (1, 0, 0)],
        ),
        (
            {
                "boxes": torch.Tensor([[1, 0.6, 0.9, 0.6], [0.1, 0.1, 0.3, 0.4]]),
                "labels": torch.Tensor([[1], [0]]),
            },
            torch.Tensor([[0.1, 0.1], [0.2, 0.3]]),
            [2, 2, 2, 6],
            [(1, 0, 1), (1, 0, 0)],
        ),
    ],
)
def test_grid_target_generation(
    annotations: Dict[str, torch.Tensor],
    anchors: torch.Tensor,
    grid_size,
    expected_indexes,
):
    target = generate_target_from_anotation(annotations, grid_size, anchors)
    print(target)
    for i, idx in enumerate(expected_indexes):
        print(idx)
        assert (target[idx][0:4] == annotations["boxes"][i]).all()
        assert target[idx][4]
        assert (target[idx][5:] == annotations["labels"][i]).all()


# def test_iou_computation():
#
#     annotations = {
#         "boxes": torch.Tensor([[0.1, 0.1, 0.1, 0.1]]),
#         "labels": torch.Tensor([[1, 0]]),
#     }
#
#     grid_size = [1, 10, 10, 7]
#
#     det = torch.zeros(grid_size, dtype=torch.float32)
#
#     det[0, 0, 0, 0:4] = torch.Tensor([[0.09, 0.09, 0.09, 0.09]])
#
#     out = generate_target_from_anotation(
#         annotations, grid_size, torch.Tensor([[0.1, 0.1]])
#     )
#
#     ious = box_iou(det, out)
#     assert ious.sum().item() == .0
