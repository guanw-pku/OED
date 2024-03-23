import os
import sys
import random

def run_train(device='0', tag='spatial_predcls'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --output_dir exps/{tag}/ \
                    --dataset_file ag_single \
                    --ag_path data/action-genome \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 3 \
                    --dec_layers_interaction 3 \
                    --batch_size 2 \
                    --epochs 12 \
                    --lr_drop 9 \
                    --pretrained exps/params/detr/detr-r50-pre-2stage_e29.pth \
                    --dsgg_task predcls")
    


if __name__ == "__main__":
    device = '0,1,2,3,4,5,6,7'
    run_train(device=device, tag='spatial_predcls')
