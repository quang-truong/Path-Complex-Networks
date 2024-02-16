#!/bin/bash

python -m tools.train_eval \
  --batch_size 128 \
  --complex_type path \
  --conv_type B \
  --dataset MOLHIV \
  --device 1 \
  --drop_position lin2 \
  --drop_rate 0.0 \
  --emb_dim 64 \
  --epochs 150 \
  --eval_metric ogbg-molhiv \
  --exp_name pcn-molhiv \
  --final_readout sum \
  --graph_norm bn \
  --indrop_rate 0.0 \
  --init_method sum \
  --jump_mode None \
  --lr 0.0001 \
  --lr_scheduler None \
  --max_dim 2 \
  --model ogb_embed_sparse_cin \
  --nonlinearity relu \
  --num_fc_layers 2 \
  --num_layers 1 \
  --num_layers_combine 1 \
  --num_layers_update 2 \
  --num_workers 0 \
  --preproc_jobs 32 \
  --readout mean \
  --start_seed 0 \
  --stop_seed 9 \
  --task_type bin_classification \
  --train_eval_period 10 \
  --use_coboundaries True \
  --use_edge_features \
  --disable_graph_norm True
