#!/bin/bash

python -m tools.train_eval \
  --batch_size 128 \
  --complex_type path \
  --conv_type B \
  --dataset NCI1 \
  --device 0 \
  --drop_position lin2 \
  --drop_rate 0 \
  --emb_dim 32 \
  --epochs 150 \
  --eval_metric accuracy \
  --exp_name pcn-nci1 \
  --final_readout sum \
  --folds 10 \
  --graph_norm bn \
  --init_method sum \
  --jump_mode None \
  --lr 0.0005 \
  --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.2 \
  --lr_scheduler_decay_steps 60 \
  --max_dim 7 \
  --model sparse_cin \
  --nonlinearity relu \
  --num_fc_layers 2 \
  --num_layers 2 \
  --num_layers_combine 1 \
  --num_layers_update 2 \
  --preproc_jobs 32 \
  --readout sum \
  --task_type classification \
  --train_eval_period 50
  