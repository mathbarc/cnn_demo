# cd datasets
# dvc pull coco_dataset
# cd ..
export DATASET_PATH="$PWD/datasets/"
python train_object_detection.py
