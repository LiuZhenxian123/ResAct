import numpy as np
import torch
import argparse
import os
import gym
import time
import json
import dmc2gym

import utils
from logger import Logger
from video import VideoRecorder

from carla_env import CarlaEnv
from resact_agent import ResActAgent
from torchvision import transforms
import data_augs as rad
from data_augs import random_translate

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='resact', type=str, choices=['resact'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=20, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    #rad
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--image_size', default=108, type=int)
    parser.add_argument('--data_augs', default='translate', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    
    #resact
    parser.add_argument('--save_action', default=False, action='store_true') #record actions during eval
    parser.add_argument('--save_embedding', default=False, action='store_true') #record latent embeddings for visualization
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step,episode, args,device=None, embed_viz_dir=None, do_carla_metrics=None,record_eval_action=False):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0

    # record embeddings for t-SNE visualization
    obses = []
    values = []
    embeddings = []
    action_list = []
    prev_action_list = []
    actions_log_path = args.domain_name + '-'+ args.task_name+ '-' +'seed'+str(args.seed)+'-'+'actions.log'
    with open(actions_log_path, 'a') as actions_log:
        for i in range(num_episodes):
            # carla metrics:
            dist_driven_this_episode = 0.

            prev_obs = None
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            prev_action = None
            env_action_sample = env.action_space.sample()
            current_rollout_step = 0

            while not done:
                # center crop image
                if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                    obs = utils.center_crop_image(obs,args.image_size)
                if args.encoder_type == 'pixel' and 'translate' in args.data_augs:
                    # first crop the center with pre_image_size
                    obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                    # then translate cropped to center
                    obs = utils.center_translate(obs, args.image_size)
                with utils.eval_mode(agent):
                    prev_obs = np.zeros_like(obs) if prev_obs is None else prev_obs
                    prev_action = np.zeros_like(env_action_sample) if prev_action is None else prev_action
                    action = agent.select_action(prev_obs,obs,prev_action)
                    current_rollout_step += 1
                    action_distance = float(np.linalg.norm(prev_action - action))
                #record actions for MAD caculations
                if record_eval_action and not np.all(prev_action == 0):
                    log_data = {
                        "step": step,
                        "episode": episode,
                        "prev_action": prev_action.tolist(),
                        "action": action.tolist(),
                        "action_distance": "{:.2f}".format(action_distance),
                        "rollout_episode": i + 1,
                        "rollout_step": current_rollout_step
                    }
                    actions_log.write(json.dumps(log_data) + '\n')
                #record embeddings for t-SNE visualization
                if embed_viz_dir:
                    obses.append(obs)
                    action_list.append(action)
                    prev_action_list.append(prev_action)
                    with torch.no_grad():
                        values.append(min(agent.critic(torch.Tensor(prev_obs).to(device).unsqueeze(0), torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                        embeddings.append(agent.critic.encoder(torch.Tensor(prev_obs).unsqueeze(0).to(device),torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())
                prev_obs = obs
                prev_action = action
                obs, reward, done, info = env.step(action)

                # metrics:
                if do_carla_metrics:
                    dist_driven_this_episode += info['distance']
                    crash_intensity += info['crash_intensity']
                    steer += abs(info['steer'])
                    brake += info['brake']
                    count += 1

                video.record(env)
                episode_reward += reward
            # metrics:
            if do_carla_metrics:
                reason_each_episode_ended.append(info['reason_episode_ended'])
                distance_driven_each_episode.append(dist_driven_this_episode)

            video.save('%d.mp4' % step)
            L.log('eval/episode_reward', episode_reward, step)

    if embed_viz_dir and step>=100000 and step%20000==0:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings,'actions':action_list, 'prev_actions':prev_action_list}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'resact':
        agent = ResActAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            data_augs=args.data_augs
        )

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = args.pre_transform_image_size if 'crop' in args.data_augs else args.image_size
    pre_image_size = args.pre_transform_image_size # record the pre transform image size for translation

    if args.domain_name == 'carla':
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?

        eval_env = env
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat,
            camera_id=args.camera_id,
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat,
            camera_id=args.camera_id,
        )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,pre_transform_image_size,pre_transform_image_size)

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        pre_image_size=pre_image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                if  args.save_action:
                    record_eval_action = (episode % 20 == 0)
                else:
                    record_eval_action = False
                if args.save_embedding:
                    embed_viz_dir = 'vis'
                    if not os.path.exists(embed_viz_dir):
                        os.makedirs(embed_viz_dir)
                else:
                    embed_viz_dir = None
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step,episode,args,device=device,embed_viz_dir=embed_viz_dir,record_eval_action=record_eval_action)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            prev_obs = None
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0
            prev_action = None
            first_episode_step = True
            env_action_sample = env.action_space.sample()

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                prev_obs = np.zeros_like(obs) if prev_obs is None else prev_obs
                prev_action = np.zeros_like(env_action_sample) if prev_action is None else prev_action
                action = agent.sample_action(prev_obs,obs,prev_action)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        if prev_action is None:
            replay_buffer.add(np.zeros_like(obs),obs, action,np.zeros_like(action), curr_reward, reward, next_obs, done_bool)
        else:
            replay_buffer.add(prev_obs,obs, action,prev_action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        prev_obs = obs
        prev_action = action

        obs = next_obs
        episode_step += 1


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    main()
