# Visual Reinforcement Learning with Residual Action

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```shell
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```shell
conda activate resact
```
## Instructions
To train a ResAct agent on the `cheetah run` task from image-based observations run `./run_cheetah.sh` from the root of this directory. The `./run_cheetah.sh` file contains the following command, which you can modify to try different environments / hyperparamters.

```shell
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cheetah \
    --task_name run \
    --decoder_type identity \
    --encoder_type pixel --work_dir ./logs/cheetah \
    --action_repeat 4 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 108 \
    --agent resact --frame_stack 3 --data_augs translate \
    --seed 1 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 20 --batch_size 128 \
    --num_train_steps 200000
```

## Logging 

In your console, you should see printouts that look like this:

```shell
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
The above output decodes as:

```shell
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if it is trained from pixels and decoder)
```
while an evaluation entry:
```shell
| eval | S: 0 | ER: 0
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```shell
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh.
## CARLA
Download CARLA from https://github.com/carla-simulator/carla/releases, e.g.:
(0.9.6) http://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.6.tar.gz

Add to your python path:
```shell
export PYTHONPATH=$PYTHONPATH:/path_to_carla/carla/PythonAPI
export PYTHONPATH=$PYTHONPATH:/path_to_carla/carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/path_to_carla/carla/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```
Start CARLA simulator in terminal 1:
```shell
cd path_to_carla
CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -graphicsadapter=0 -carla-rpc-port=10000 -opengl -RenderOffScreen
```
Run training script in terminal 2:
```shell
CUDA_VISIBLE_DEVICES=0 ./run_carla.sh --agent resact --domain_name carla --port 10000
```