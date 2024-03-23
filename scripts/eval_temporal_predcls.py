import os
import sys
import random

def run_eval(device='0', ckpt_path='exps/params/predcls/temporal/checkpoint_5.pth'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --dataset_file ag_multi \
                    --ag_path data/action-genome \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 3 \
                    --dec_layers_interaction 3 \
                    --eval \
                    --batch_size 1 \
                    --num_workers 0 \
                    --freeze_mode 1 \
                    --query_temporal_interaction \
                    --pretrained {ckpt_path} \
                    --dsgg_task predcls")


if __name__ == "__main__":
    device = '0'
    run_eval(device=device, ckpt_path='exps/params/predcls/temporal/checkpoint_5.pth')
