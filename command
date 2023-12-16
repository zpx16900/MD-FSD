python fsod_train_net.py --num-gpus 1 --config-file configs/fsod/finetune_R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log.txt
python fsod_train_net.py --resume output/fsod/R_50_C4_1x/model_0089999.pth --num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml
python fsod_train_net.py --num-gpus 1 --config-file configs/fsod/finetune_R_50_C4_1x.yaml


train
python fsod_train_net.py --num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml

eval
python fsod_train_net.py --num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth
model_0209999
python fsod_train_net.py --num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_0139999.pth


--resume --num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml

--num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_0039999.pth

--num-gpus 1 --config-file configs/fsod/R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth