#!/bin/bash

python -m tools.train_eval \
  --batch_size 128 \
  --complex_type path \
  --conv_type A \
  --dataset PROTEINS \
  --device 0 \
  --drop_position lin2 \
  --drop_rate 0 \
  --emb_dim 16 \
  --epochs 200 \
  --eval_metric accuracy \
  --exp_name pcn-proteins \
  --final_readout sum \
  --folds 10 \
  --graph_norm bn \
  --init_method sum \
  --jump_mode None \
  --lr 0.005 \
  --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.6 \
  --lr_scheduler_decay_steps 40 \
  --max_dim 2 \
  --model sparse_cin \
  --nonlinearity relu \
  --num_fc_layers 3 \
  --num_layers 3 \
  --num_layers_combine 1 \
  --num_layers_update 2 \
  --preproc_jobs 32 \
  --readout sum \
  --task_type classification \
  --train_eval_period 50 \
  