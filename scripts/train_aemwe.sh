#!/bin/bash

[ ! -d "exp_res" ] && mkdir -p exp_res

MODEL_ID="aemwe_loocv"

nohup python -u train.py \
  --task train \
  --root_path ./data/aemwe_all_p0.1 \
  --model_id $MODEL_ID \
  --model ElectroTS \
  --time_cutoff 150 \
  --seq_len 1000 \
  --label_len 125 \
  --pred_len 250 \
  --e_layers 4 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 400 \
  --d_core 400 \
  --d_ff 400 \
  --learning_rate 0.0001 \
  --lradj cosine \
  --train_epochs 20 \
  --patience 20 \
  --batch_size 32 \
  --save_model > exp_res/$MODEL_ID.log 2>&1 &
