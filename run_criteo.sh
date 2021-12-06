#!/usr/bin/env bash
DATASET=criteo
SEED=42
RES_MODE='last_layer'
SCALE_ATT='False'
RELU_BEFORE_ATT='False'
DROPOUT=0.2
WEIGHT_DECAY=5e-6
BASE_MODEL='DSelfAttnWgt'
NUM_LAYERS=5
EMBED_DIM=32
MAGIC='False'
DEVICE='cuda'

python main.py \
  --seed $SEED \
  --dataset $DATASET \
  --res_mode $RES_MODE \
  --scale_att $SCALE_ATT \
  --relu_before_att $RELU_BEFORE_ATT \
  --dropout $DROPOUT \
  --weight_decay $WEIGHT_DECAY \
  --base_model $BASE_MODEL \
  --num_layers $NUM_LAYERS \
  --embed_dim $EMBED_DIM \
  --magic $MAGIC \
  --device $DEVICE
