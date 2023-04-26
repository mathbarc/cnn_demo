find . -name "*.jpeg" > dataset.txt
awk -f ../../scripts/split.awk dataset.txt
mv dataset.txt_train train.txt
mv dataset.txt_test test.txt

awk -f ../../scripts/split.awk train.txt
mv train.txt_train train.txt
mv train.txt_test valid.txt
