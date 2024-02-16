#!/bin/bash

python -m tools.train_eval \
  --batch_size 128 \
  --complex_type path \
  --conv_type B \
  --dataset NCI109 \
  --device 0 \
  --drop_position lin2 \
  --drop_rate 0 \
  --emb_dim 64 \
  --epochs 200 \
  --eval_metric accuracy \
  --exp_name pcn-nci109 \
  --final_readout sum \
  --folds 10 \
  --graph_norm bn \
  --init_method mean \
  --jump_mode cat \
  --lr 0.0005 \
  --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.4 \
  --lr_scheduler_decay_steps 20 \
  --max_dim 3 \
  --model sparse_cin \
  --nonlinearity relu \
  --num_fc_layers 2 \
  --num_layers 7 \
  --num_layers_combine 1 \
  --num_layers_update 1 \
  --preproc_jobs 32 \
  --readout sum \
  --task_type classification \
  --train_eval_period 50
  