#!/bin/bash

DOMAIN=carla096
TASK=highway
AGENT=resact
SEED=1
DECODER_TYPE=identity
TRANSITION_MODEL=deterministic

SAVEDIR=./log-${DOMAIN}-${TASK}-${AGENT}-seed${SEED}
mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent ${AGENT} \
    --encoder_type pixelCarla096 \
    --decoder_type ${DECODER_TYPE} \
    --action_repeat 4 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --encoder_stride 2 \
    --hidden_dim 1024 \
    --replay_buffer_capacity 100000 \
    --total_frames 10000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR} \
    --transition_model_type ${TRANSITION_MODEL} \
    --seed ${SEED} $@ \
    --frame_stack 3 \
    --image_size 84 \
    --data_augs rand_conv \
    --num_eval_episodes 10 \
    --eval_freq 20 \
    >> ${SAVEDIR}/output.txt
