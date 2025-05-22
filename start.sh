cd datasets
dvc pull coco_dataset
cd ..
export DATASET_PATH="$PWD/datasets/coco_dataset"
python train_object_detection.py
