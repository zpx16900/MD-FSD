DATA_ROOT=/Volumes/home/Drive/Drive/SCI/2/codes/FewX-master/datasets/coco

cd coco

#ln -s $DATA_ROOT\\train2017 ./
#ln -s $DATA_ROOT\\val2017 ./
#ln -s $DATA_ROOT\\annotations ./

python 1_split_filter.py ./
#python3 2_balance.py ./
python 3_gen_support_pool.py ./
python 4_gen_support_pool_10_shot.py ./

