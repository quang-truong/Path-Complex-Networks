#!/bin/bash

python -m tools.train_eval \
  --start_seed 0 \
  --stop_seed 9 \
  --device 0 \
  --dataset SR-GRAPHS \
  --complex_type path \
  --exp_name pcn-sr \
  --model sparse_cin \
  --use_coboundaries True \
  --drop_rate 0.0 \
  --graph_norm id \
  --nonlinearity elu \
  --readout sum \
  --final_readout sum \
  --lr_scheduler None \
  --num_layers 6 \
  --num_layers_update 1 \
  --num_layers_combine 1 \
  --emb_dim 16 \
  --batch_size 8 \
  --num_workers 8 \
  --task_type isomorphism \
  --eval_metric isomorphism \
  --max_dim 3 \
  --init_method sum \
  --preproc_jobs 64 \
  --note "pcn-sr num_classes 32" \
  --untrained
  