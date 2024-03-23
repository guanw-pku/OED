import os
import sys
import random

def run_train(device='0', tag='temporal_predcls'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --pretrained exps/params/predcls/spatial/checkpoint_9.pth \
                    --output_dir exps/{tag}/ \
                    --dataset_file ag_multi \
                    --ag_path data/action-genome \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 3 \
                    --dec_layers_interaction 3 \
                    --epochs 6 \
                    --lr_drop 3 \
                    --batch_size 2 \
                    --num_workers 0 \
                    --freeze_mode 1 \
                    --query_temporal_interaction \
                    --dsgg_task predcls")
    


if __name__ == "__main__":
    device = '0,1,2,3,4,5,6,7'
    run_train(device=device, tag='temporal_predcls')
