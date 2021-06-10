import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from sprites_env.envs import sprites
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from a2c_ppo_acktr.model import Encoder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from model import *
from sprites_env.envs import sprites
from dataset import *

def train(model, batch, optimizer, decoder_optimizer, device):
    avg_loss = 0.0
    avg_decoded_loss = 0.0

    # for obs, reward_targets in zip(batch['obs'], batch['rewards']):
    for obs, agent_x, agent_y, target_x, target_y in zip(batch['obs'], batch['agent_x'], batch['agent_y'], batch['target_x'], batch['target_y']):
        optimizer.zero_grad()
        obs, agent_x, agent_y, target_x, target_y = obs.to(device), agent_x.to(device), agent_y.to(device), target_x.to(device), target_y.to(device)
        reward_targets = torch.stack((agent_x, agent_y, target_x, target_y))
        reward_predicted = model(obs).squeeze()
        loss = model.criterion(reward_predicted, reward_targets)
        avg_loss += loss

        # loss.backward()
        # optimizer.step()
    
    avg_loss.backward(retain_graph=True)
    optimizer.step()

    for obs in batch['obs']:
        decoder_optimizer.zero_grad()
        encoded_img = model.encoder(obs[-1][None, None, :].detach().clone().to(device)).to(device)
        decoded_img = model.decoder(encoded_img).squeeze().to(device)
        decoded_loss = model.criterion(decoded_img, obs[-1].to(device))
        avg_decoded_loss += decoded_loss
                
        # decoded_loss.backward()
        # decoder_optimizer.step()

    avg_decoded_loss.backward()
    decoder_optimizer.step()

    l = len(batch['obs'])
    avg_loss = avg_loss / l
    avg_decoded_loss = avg_decoded_loss / l

    return avg_loss.item(), decoded_img[None, :], avg_decoded_loss.item()


def main():
    args = get_args()
    log_dir = 'runs3/env=' + args.env_name + '_lr=' + str(args.lr) + '_num_steps' + str(args.num_steps) + ' ||' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if not(os.path.exists(log_dir)):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    f = args.conditioning_frames
    t = args.time_steps
    assert t > f

    # load data
    dl, traj_images, ground_truth = dataloader(
        args.image_resolution, t, args.batch_size, f, args.reward, args.dataset_length)

    traj_images = traj_images.to(device)

    model = Model(t, f+1, args.tasks, args.image_resolution, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=args.learning_rate)
    train_loss = []
    train_decoded_loss = []

    for epoch in range(args.num_epochs-1):
        running_loss = 0.0
        running_decoded_loss = 0.0
        num_batch = 0
        for batch in dl:
            loss, decoded_img, decoded_loss = train(
                model, batch, optimizer, decoder_optimizer, device)
            running_loss += loss
            running_decoded_loss += decoded_loss
            num_batch += 1

        # print or store data
        running_loss = running_loss / num_batch
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, running_loss))
        train_loss.append(running_loss)

        running_decoded_loss = running_decoded_loss / num_batch
        train_decoded_loss.append(running_decoded_loss)

        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('Loss/decoded', running_decoded_loss, epoch)

        if epoch % 5 == 0:
            decoded_img = decoded_img * 255.0
            writer.add_image('decoded_epoch{}'.format(
                epoch), decoded_img.to(torch.uint8))

    # decode and generate images with respect to reward functions
    output = model.test_decode(traj_images)
    output = output * 255

    img = make_image_seq_strip([output[None, :, None].repeat(
        3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    writer.add_image('ground_truth', ground_truth)
    writer.add_image('test_decoded', img[0])

    # RL
    # load_dir = './trained_models/'
    # load_path = os.path.join(load_dir, 'Sprites-v0') # use v0's encoder
    # encoder = Encoder()
    # encoder.load_state_dict(torch.load(os.path.join(load_path, 'seed=' + str(args.seed) + ".pt")))
    # trained_encoder = copy.deepcopy(encoder).to(device)
    
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy},
        # trained_encoder=trained_encoder
        trained_encoder=model.encoder
        )
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    vid_rollouts = RolloutStorage(128, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    vid_rollouts.obs[0].copy_(obs)
    vid_rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        # decide if videos should be rendered/logged at this iteration
        if j % 10 == 0:  # video_log_freq, a hyperparameter
            log_video = True
            batch_image_obs = []
        else:
            log_video = False

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            if log_video:
                batch_image_obs.append(envs.render())

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if log_video:
            # perform logging
            # if j == num_updates-1:
            #   for step in range(128):  # num steps for video collection               
            #       with torch.no_grad():
            #           value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
            #               vid_rollouts.obs[step], vid_rollouts.recurrent_hidden_states[step], vid_rollouts.masks[step])
            #           batch_image_obs.append(envs.render())
            #           obs, reward, done, infos = envs.step(action)              
            #           for info in infos:
            #             if 'episode' in info.keys():
            #                 episode_rewards.append(info['episode']['r'])

            #           # If done then clean the history of observations.
            #           masks = torch.FloatTensor(
            #               [[0.0] if done_ else [1.0] for done_ in done])
            #           bad_masks = torch.FloatTensor(
            #               [[0.0] if 'bad_transition' in info.keys() else [1.0]
            #               for info in infos])
            #           vid_rollouts.insert(obs, recurrent_hidden_states, action,
            #                       action_log_prob, value, reward, masks, bad_masks)

            # Need [N, T, C, H, W] input tensor for video logging
            video = torch.tensor(batch_image_obs).unsqueeze(0).unsqueeze(2)
            writer.add_video('{}'.format('train_rollouts'), video, j, fps=10)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + "_steps=" + str(args.num_steps) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            writer.add_scalar('Average Return', np.mean(episode_rewards), total_num_steps)
            writer.add_scalar('Value Loss', value_loss, total_num_steps)
            writer.add_scalar('Action Loss', action_loss, total_num_steps)
            writer.add_scalar('Dist Entropy', dist_entropy, total_num_steps)                        

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
    writer.flush()


if __name__ == "__main__":
    main()
