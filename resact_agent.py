# Copyright (c) Facebook, Inc. and its affiliates.weight_init
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sac_ae import  Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder
import data_augs as rad


class DeepMDPAgent(object):
    """Baseline algorithm with transition model and various decoder types."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        transition_model_type='deterministic',
        num_layers=4,
        num_filters=32,
        data_augs=''
    ):
        self.reconstruction = False
        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction =  True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type
        self.data_augs = data_augs
        self.image_size = obs_shape[-1]
        self.augs_funcs = {}

        aug_to_func = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        decoder_params = list(self.transition_model.parameters())  + list(self.reward_decoder.parameters())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)
            decoder_params += list(self.decoder.parameters())

        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, prev_obs,obs,prev_action):
        with torch.no_grad():
            prev_obs = torch.FloatTensor(prev_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
            prev_obs = prev_obs.unsqueeze(0)
            obs = obs.unsqueeze(0)
            prev_action = torch.from_numpy(prev_action).reshape(1,-1).to(self.device)
            mu, _, _, _ = self.actor(
                prev_obs,obs,prev_action, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()
    
    # def select_action_2(self, obs,prev_act,step):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).to(self.device)
    #         obs = obs.unsqueeze(0)
    #         mu, _, _, _ = self.actor(
    #             obs, compute_pi=False, compute_log_pi=False
    #         )
    #         #reshape(x,y) 需要根据任务的action space设定 carla(1,2) cheetah(1,6) finger(1,2)
    #         prev_mu = torch.from_numpy(prev_act).reshape(1,6).to(self.device)
    #         curr_q1,curr_q2 = self.critic(obs,mu)
    #         curr_q = min(curr_q1,curr_q2)
    #         prev_q1,prev_q2 = self.critic(obs,prev_mu)
    #         prev_q = min(prev_q1,prev_q2)
    #         if prev_q>curr_q:
    #             ratio = min(0.3,5000.0/step)
    #             #ratio = 0.5
    #             return (prev_mu*ratio+mu*(1-ratio)).cpu().data.numpy().flatten()
    #         else:
    #             return mu.cpu().data.numpy().flatten()


    def sample_action(self,prev_obs, obs,prev_action):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
            prev_obs = utils.center_crop_image(prev_obs, self.image_size)
         # center crop image
        # if self.encoder_type == 'pixel' and 'crop' in self.data_augs:
        #     obs = utils.center_crop_image(obs,self.image_size)
        # if self.encoder_type == 'pixel' and 'translate' in self.data_augs:
        #     # first crop the center with pre_image_size
        #     obs = utils.center_crop_image(obs, self.pre_transform_image_size)
        #     # then translate cropped to center
        #     obs = utils.center_translate(obs, self.image_size)
        with torch.no_grad():
            prev_obs = torch.FloatTensor(prev_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
            prev_obs = prev_obs.unsqueeze(0)
            obs = obs.unsqueeze(0)
            prev_action = torch.from_numpy(prev_action).reshape(1,-1).to(self.device)
            mu, pi, _, _ = self.actor(prev_obs,obs,prev_action, compute_log_pi=False)
            # print("pi is:",pi)
            # print("pi's shape:",pi.shape)
            return pi.cpu().data.numpy().flatten()
        
    def sample_action_2(self,prev_obs,obs,prev_act,step):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
            prev_obs = utils.center_crop_image(prev_obs, self.image_size)
        with torch.no_grad():
            prev_obs = torch.FloatTensor(prev_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
            prev_obs = prev_obs.unsqueeze(0)
            obs = obs.unsqueeze(0)
            prev_pi = torch.from_numpy(prev_act).reshape(1,-1).to(self.device)
            mu, pi, _, _ = self.actor(prev_obs,obs,prev_pi,compute_log_pi=False)
            # reshape(x,y) 需要根据任务的action space设定 carla(1,2) cheetah(1,6) finger(1,2)
            # prev_pi = torch.from_numpy(prev_act).reshape(1,-1).to(self.device)
            curr_q1,curr_q2 = self.critic(prev_obs,obs,pi)
            curr_q = min(curr_q1,curr_q2)
            prev_q1,prev_q2 = self.critic(prev_obs,obs,prev_pi)
            prev_q = min(prev_q1,prev_q2)
            if prev_q>curr_q:
                ratio = min(0.3,5000.0/step)
                #ratio = 0.5
                return (prev_pi*ratio+pi*(1-ratio)).cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten()

    def update_critic(self, prev_obs,obs, action,reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(obs,next_obs,action)
            target_Q1, target_Q2 = self.critic_target(obs,next_obs, policy_action)
            target_V = torch.min(target_Q1, 
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(prev_obs,obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, prev_obs,obs,prev_action, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(prev_obs,obs,prev_action, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(prev_obs,obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_transition_reward_model(self, prev_obs,obs, action, next_obs, reward, L, step):
        h = self.critic.encoder(prev_obs,obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(obs,next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_decoder(self, prev_obs,obs, action, target_obs, L, step):  #  uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(prev_obs,obs)
        if not self.reconstruction:
            next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = utils.preprocess_obs(target_obs)
            rec_obs = self.decoder(next_h)
            loss = F.mse_loss(target_obs, rec_obs)
        else:
            rec_obs = self.decoder(h)
            loss = F.mse_loss(obs, rec_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        prev_obs,obs, action,prev_action, _, reward, next_obs, not_done = replay_buffer.sample(self.augs_funcs)

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(prev_obs,obs, action, reward, next_obs, not_done, L, step)
        self.update_transition_reward_model(prev_obs,obs, action, next_obs, reward, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(prev_obs,obs, prev_action,L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            self.update_decoder(prev_obs,obs, action, next_obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
