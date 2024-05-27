DOMAIN_NAME=cheetah
TASK_NAME=run
AGENT=resact
SEED=12

SAVEDIR=./log-${DOMAIN_NAME}-${TASK_NAME}-${AGENT}-seed${SEED}
mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name ${DOMAIN_NAME} \
    --agent ${AGENT} \
    --task_name ${TASK_NAME} \
    --encoder_type pixel \
    --decoder_type identity \
    --init_steps 10000 \
    --num_train_steps 200000 \
    --batch_size 128 \
    --replay_buffer_capacity 100000 \
    --num_filters 32 \
    --hidden_dim 1024 \
    --action_repeat 4 \
    --save_video \
    --work_dir ${SAVEDIR} \
    --critic_lr 2e-4 \
    --actor_lr 2e-4 \
    --encoder_lr 2e-4 \
    --alpha_lr 1e-4 \
    --encoder_feature_dim 64 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --alpha_beta 0.5 \
    --init_temperature 0.1 \
    --seed ${SEED} \
    --pre_transform_image_size 100 \
    --image_size 108 \
    --data_augs translate \
    >> ${SAVEDIR}/output.txt \
