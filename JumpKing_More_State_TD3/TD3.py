#!/usr/env/bin python
#
# Game Screen
#

import pygame
import sys
import os
import inspect
import pickle
import numpy as np
import argparse
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus

from Start import Start

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate more states according to the window around the king
def Get_states_big(semantic_ref_array, cur_level, cur_rect_x, cur_rect_y):
	# Init
	level_selected = cur_level
	block_w = 20  # 1*rect_width
	block_h = 24  # 1*rect_height
	num_blocks = 133  # 133 blocks in total
	window_w = 19*block_w  # 380 pixs
	window_h = 7*block_h  # 168 pixs

	# 1 Padding
	temp_array = semantic_ref_array[level_selected]
	origin_c = temp_array.shape[0]
	origin_h = temp_array.shape[1]
	origin_w = temp_array.shape[2]
	padded_h = origin_h + 5*block_h + block_h
	padded_w = origin_w + 2*9*block_w
	array_padded = np.ones((origin_c, padded_h, padded_w))
	# print('padded: ', array_padded.shape)
	array_padded[0:origin_c, (5*block_h):(5*block_h + origin_h), (9*block_w):(9*block_w + origin_w)] = temp_array[:, :, :]
	# matplotlib.image.imsave('Temp/test_padding.png', array_padded.transpose(1, 2, 0))

	# Get states per block - 133 blocks
	LT_x = round(cur_rect_x) + 9*block_w
	LT_y = round(cur_rect_y) + 5*block_h + int(origin_h/2)
	block_storage = np.zeros((num_blocks, block_h, block_w))
	for i in range(num_blocks):
		temp_row = int(i/19)
		temp_col = i%19
		cropped_h0 = LT_y - 5*block_h + temp_row*block_h
		cropped_hf = LT_y - 5*block_h + temp_row*block_h + block_h
		cropped_w0 = LT_x - 9*block_w + temp_col*block_w
		cropped_wf = LT_x - 9*block_w + temp_col*block_w + block_w
		block_storage[i, :, :] = array_padded[0, cropped_h0:cropped_hf, cropped_w0:cropped_wf]

	# print('rect_x: ', round(cur_rect_x))
	# print('rect_y: ', round(cur_rect_y))
	# print('LT_x: ', LT_x)
	# print('LT_y: ', LT_y)
	# for i in range(num_blocks):
	# 	print('block_{}'.format(i), block_storage[i].shape)
	# matplotlib.image.imsave('Temp/test_window_check.png', array_padded[:, (LT_y - 5*block_h):(LT_y + 2*block_h), (LT_x - 9*block_w):(LT_x + 10*block_w)].transpose(1, 2, 0))

	output_states_array = np.zeros((7, 19))
	output_states_list = []
	for i in range(num_blocks):
		temp_row = int(i / 19)
		temp_col = i % 19
		flag_collision = (int(np.sum(block_storage[i])) != 0)
		output_states_array[temp_row, temp_col] = flag_collision
		if i != 104:
			output_states_list.append(flag_collision)
	# print(len(output_states_list))


	return (output_states_array, output_states_list)


# Generate more states according to the window around the king
def Get_states(semantic_ref_array, cur_level, cur_rect_x, cur_rect_y):
	# Init
	level_selected = cur_level
	block_w = 60  # 3*rect_width
	block_h = 24  # 1*rect_height
	window_w = 3*block_w
	window_h = 3*block_h

	# 1 Padding
	temp_array = semantic_ref_array[level_selected]
	origin_c = temp_array.shape[0]
	origin_h = temp_array.shape[1]
	origin_w = temp_array.shape[2]
	padded_h = origin_h + 2*block_h
	padded_w = origin_w + 2*(block_w + int(block_w/3))
	array_padded = np.ones((origin_c, padded_h, padded_w))
	# print('padded: ', array_padded.shape)
	array_padded[0:origin_c, block_h:(block_h + origin_h), (block_w + int(block_w/3)):(block_w + int(block_w/3) + origin_w)] = temp_array[:, :, :]
	#matplotlib.image.imsave('Temp/test_padding.png', array_padded.transpose(1, 2, 0))

	# Get states per block
	LT_x = round(cur_rect_x) + block_w + int(block_w/3)
	LT_y = round(cur_rect_y) + block_h + int(origin_h/2)
	block_1 = array_padded[0, (LT_y - block_h):LT_y, (LT_x - (block_w + int(block_w/3))):(LT_x - int(block_w/3))]
	block_2 = array_padded[0, (LT_y - block_h):LT_y, (LT_x - int(block_w/3)):(LT_x + int(2*block_w/3))]
	block_3 = array_padded[0, (LT_y - block_h):LT_y, (LT_x + int(2*block_w/3)):(LT_x + int(5*block_w/3))]
	block_4 = array_padded[0, LT_y:(LT_y + block_h), (LT_x - (block_w + int(block_w/3))):(LT_x - int(block_w/3))]
	block_5 = array_padded[0, LT_y:(LT_y + block_h), (LT_x - int(block_w/3)):(LT_x + int(2*block_w/3))]
	block_6 = array_padded[0, LT_y:(LT_y + block_h), (LT_x + int(2*block_w/3)):(LT_x + int(5*block_w/3))]
	block_7 = array_padded[0, (LT_y + block_h):(LT_y + 2*block_h), (LT_x - (block_w + int(block_w/3))):(LT_x - int(block_w/3))]
	block_8 = array_padded[0, (LT_y + block_h):(LT_y + 2*block_h), (LT_x - int(block_w/3)):(LT_x + int(2*block_w/3))]
	block_9 = array_padded[0, (LT_y + block_h):(LT_y + 2*block_h), (LT_x + int(2 * block_w / 3)):(LT_x + int(5 * block_w / 3))]
	# print('rect_x: ', round(cur_rect_x))
	# print('rect_y: ', round(cur_rect_y))
	# print('LT_x: ', LT_x)
	# print('LT_y: ', LT_y)
	# print('1', block_1.shape)
	# print('2', block_2.shape)
	# print('3', block_3.shape)
	# print('4', block_4.shape)
	# print('6', block_6.shape)
	# print('7', block_7.shape)
	# print('8', block_8.shape)
	# print('9', block_9.shape)
	#matplotlib.image.imsave('Temp/test_block_check.png', array_padded[:, LT_y:(LT_y + block_h), (LT_x - int(block_w/3)):(LT_x + int(2*block_w/3))].transpose(1, 2, 0))

	output_states = np.zeros((3, 3))
	# print('sum1: ', np.sum(block_1))
	# print('sum2: ', np.sum(block_2))
	# print('sum3: ', np.sum(block_3))
	# print('sum4: ', np.sum(block_4))
	# print('sum6: ', np.sum(block_6))
	# print('sum7: ', np.sum(block_7))
	# print('sum8: ', np.sum(block_8))
	# print('sum9: ', np.sum(block_9))
	output_states[0, 0] = (int(np.sum(block_1)) != 0)
	output_states[0, 1] = (int(np.sum(block_2)) != 0)
	output_states[0, 2] = (int(np.sum(block_3)) != 0)
	output_states[1, 0] = (int(np.sum(block_4)) != 0)
	output_states[1, 1] = (int(np.sum(block_5)) != 0)
	output_states[1, 2] = (int(np.sum(block_6)) != 0)
	output_states[2, 0] = (int(np.sum(block_7)) != 0)
	output_states[2, 1] = (int(np.sum(block_8)) != 0)
	output_states[2, 2] = (int(np.sum(block_9)) != 0)
	#print(output_states)


	return output_states


# Create 43 ref arrays
def Create_ref_array():
	# Init resized size
    resized_h = 720
    resized_w = 480
    # Reshape and resize the semantic pictures of 43 levels
    folder_path = "./Semantic_MG"
    temp_array = np.zeros((43, 3, resized_h, resized_w))
    temp_count = 0
    path_list = os.listdir(folder_path)
    path_list.sort(key=lambda x: int(x[:-4]))

    for filename in path_list:
        #print(filename)  # just for test
        # img is used to store the image data
        img = cv2.imread(folder_path + "/" + filename)
        # Combine the current level and the next level
        if temp_count >= 0 and temp_count < 42:
            next_img = cv2.imread(folder_path + "/" + path_list[temp_count + 1])
            img = np.append(next_img, img, axis=0)
        else:
            pass
        img_resized = cv2.resize(img, (resized_w, resized_h))/255
        #print(img_resized.shape)
        #matplotlib.image.imsave('Temp/test{}.png'.format(temp_count), img_resized)
        img_CHW = img_resized.transpose(2, 0, 1)  # (C, H, W)
        #img_tensor = torch.tensor(img_CHW)
        temp_array[temp_count, :, :, :] = img_CHW[:, :, :]
        temp_count += 1

    #temp_test = temp_tensor[5, :, :, :].numpy()
    #print(temp_test.shape)
    #print(temp_test)
    #temp_test = temp_test.transpose(1, 2, 0)
    #matplotlib.image.imsave('Temp/test{}.png'.format(temp_count), temp_test)


    return temp_array


#Construct Neural Networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc_units=256, fc1_units=256):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=-1)
        return out
        #return torch.tanh(self.fc3(x)) * self.max_action

# Q1-Q2-Critic Neural Network

class Critic_Q(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        super(Critic_Q, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc1_units)
        self.l5 = nn.Linear(fc1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xa = torch.cat([state, action.float()], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


class TD3:
    def __init__(
            self,
            state_size,
            action_size,
            gamma=0.99,  # discount factor
            lr_actor=3e-4,
            lr_critic=3e-4,
            batch_size=100,
            buffer_capacity=1000000,
            tau=0.02,  # soft update parameter
            random_seed=666,
            cuda=True,
            policy_noise=0.2,
            std_noise=0.1,
            noise_clip=0.5,
            policy_freq=2,  # target network update period
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.upper_bound = float(action_size - 1)  # action space upper bound
        self.lower_bound = float(0)  # action space lower bound
        self.create_actor()
        self.create_critic()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise

    def create_actor(self):
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'max_action': self.upper_bound
        }
        self.actor = Actor(**params).to(self.device)
        self.actor_target = Actor(**params).to(self.device)

    def create_critic(self):
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        self.critic = Critic_Q(**params).to(self.device)
        self.critic_target = Critic_Q(**params).to(self.device)

    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        # add samples to replay memory
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self):
        # random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def learn_and_update_weights_by_replay(self, training_iterations):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        for it in range(training_iterations):
            mini_batch = self.get_random_sample_from_replay_mem()
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)

            # Training and updating Actor & Critic networks.
            # Train Critic
            target_actions_logits = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(target_actions_logits.shape).data.normal_(0, self.policy_noise).to(self.device)

            # clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions_logits = F.softmax(target_actions_logits + offset_noises, dim=-1)
            #target_actions = torch.argmax(target_actions_logits, dim=-1, keepdim=True)

            # Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions_logits)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            # Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            # Train Actor
            # Delayed policy updates
            if it % self.policy_freq == 0:
                # Minimize the loss
                actions_logits = self.actor(state_batch)
                #actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
                actor_loss, _ = self.critic(state_batch, actions_logits)
                actor_loss = - actor_loss.mean()

                # Optimize the actor
                self.act_opt.zero_grad()
                actor_loss.backward()
                self.act_opt.step()

                # Soft update target models
                self.soft_update_target(self.critic, self.critic_target)
                self.soft_update_target(self.actor, self.actor_target)

    def soft_update_target(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def policy(self, state):
        """select action based on ACTOR"""
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions_logits = self.actor(state)
            actions = torch.argmax(actions_logits, dim=-1)
        self.actor.train()
        return actions, actions_logits.squeeze()


# SENWEI'S CODE
# class JKGame:
#     """ Overall class to manga game aspects """
#
#     def __init__(self, max_step=float('inf'), cheating_level=0):
#
#         self.cheating_level = cheating_level
#
#         self.cheating_location = {0: (230, 298, "left"), 1: (330, 245, "right"), 2: (240, 245, "right"),
#                                   3: (150, 245, "right")}
#
#         pygame.init()
#
#         self.environment = Environment()
#
#         self.clock = pygame.time.Clock()
#
#         self.fps = int(os.environ.get("fps"))
#
#         self.bg_color = (0, 0, 0)
#
#         self.screen = pygame.display.set_mode((
#                                               int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")),
#                                               int(os.environ.get("screen_height")) * int(
#                                                   os.environ.get("window_scale"))),
#                                               pygame.HWSURFACE | pygame.DOUBLEBUF)  # |pygame.SRCALPHA)
#
#         self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))),
#                                           pygame.HWSURFACE | pygame.DOUBLEBUF)  # |pygame.SRCALPHA)
#
#         self.game_screen_x = 0
#
#         pygame.display.set_icon(pygame.image.load("images/sheets/JumpKingIcon.ico"))
#
#         self.levels = Levels(self.game_screen)
#
#         self.king = King(self.game_screen, self.levels)
#
#         self.babe = Babe(self.game_screen, self.levels)
#
#         self.menus = Menus(self.game_screen, self.levels, self.king)
#
#         self.start = Start(self.game_screen, self.menus)
#
#         self.step_counter = 0
#         self.max_step = max_step
#
#         # self.abs_total_height = 43*360-144
#
#         # New
#         self.past_state_y = None
#         self.count_same_y = 0
#
#         pygame.display.set_caption('Jump King At Home XD')
#
#     def reset(self):
#         self.king.reset(self.cheating_location[self.cheating_level][0], self.cheating_location[self.cheating_level][1],
#                         self.cheating_location[self.cheating_level][2])
#         self.levels.reset(self.cheating_level)
#         os.environ["start"] = "1"
#         os.environ["gaming"] = "1"
#         os.environ["pause"] = ""
#         os.environ["active"] = "1"
#         os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
#         os.environ["session"] = "0"
#
#         self.step_counter = 0
#         done = False
#         right_state, left_state = self.compute_nearest()
#         state = [self.king.levels.current_level, self.king.x, self.king.y,
#                  self.king.jumpCount] + right_state + left_state
#
#         self.past_state_y = self.king.y
#         self.count_same_y = 1
#
#         return done, state
#
#     def move_available(self):
#         available = not self.king.isFalling \
#                     and not self.king.levels.ending \
#                     and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
#         return available
#
#     # potential-based reward shaping
#     def Phi(self, s0, s1):
#         abs_old_y = (s0[0] + 1) * 360 - s0[1]
#         abs_new_y = (s1[0] + 1) * 360 - s1[1]
#         diff_height = abs_old_y - abs_new_y
#
#         return diff_height
#
#     # SENWEI'S FCT
#     def compute_nearest(self):
#         # (x_distance, y_distance, platform_width)
#         # find the nearest 4 platforms
#         record_left_platform = []
#         record_right_platform = []
#         for level, platform_gather in enumerate([self.levels.levels[self.levels.current_level].platforms, \
#                                                  self.levels.levels[self.levels.current_level + 1].platforms]):
#             for platform in platform_gather:
#                 if platform.x - 26 <= self.king.x <= platform.x + platform.width - 6 or \
#                         level == 0 and self.king.y + 5 < platform.y - 33 or \
#                         platform.y + (1 - level) * 360 - 33 < self.king.y + 360 - 150 or \
#                         platform.height == 360:
#                     continue
#
#                 # king on the left side of the platform
#                 elif platform.x - 26 > self.king.x:
#                     record_left_platform.append(
#                         [platform.x - 26, platform.y + (1 - level) * 360 - 33, platform.width])
#
#                 # king on the right side of the platform
#                 elif platform.x + platform.width - 6 < self.king.x:
#                     record_right_platform.append(
#                         [platform.x + platform.width - 6, platform.y + (1 - level) * 360 - 33, platform.width])
#
#         if len(record_left_platform) > 0:
#             record_left_platform = np.array(record_left_platform)
#             dist = ((self.king.x - record_left_platform[:, 0]) ** 2 + (
#                     self.king.y + 360 - record_left_platform[:, 1]) ** 2) ** (1 / 2)
#             # left_state = record_left_platform[dist.argsort()][:2].reshape(-1).tolist()
#             left_state = record_left_platform[dist.argsort()][:2]
#             left_state[:, 0] = np.abs(left_state[:, 0] - self.king.x)
#             left_state[:, 1] = np.abs(self.king.y + 360 - left_state[:, 1])
#             left_state = left_state.reshape(-1).tolist()
#             if len(left_state) != 6:
#                 left_state.extend([-1, -1, -1])
#             assert len(left_state) == 6
#         else:
#             left_state = [-1] * 6
#
#         if len(record_right_platform) > 0:
#             record_right_platform = np.array(record_right_platform)
#             dist = ((self.king.x - record_right_platform[:, 0]) ** 2 + (
#                     self.king.y + 360 - record_right_platform[:, 1]) ** 2) ** (1 / 2)
#             # right_state = record_right_platform[dist.argsort()][:2].reshape(-1).tolist()
#             right_state = record_right_platform[dist.argsort()][:2]
#             right_state[:, 0] = np.abs(self.king.x - right_state[:, 0])
#             right_state[:, 1] = np.abs(self.king.y + 360 - right_state[:, 1])
#             right_state = right_state.reshape(-1).tolist()
#             if len(right_state) != 6:
#                 right_state.extend([-1, -1, -1])
#             assert len(right_state) == 6
#         else:
#             right_state = [-1] * 6
#
#         return right_state, left_state
#
#     def step(self, action):
#         s0 = [self.king.levels.current_level, self.king.y]
#
#         while True:
#             self.clock.tick(200 * self.fps)
#             self._check_events()
#             if not os.environ["pause"]:
#                 if not self.move_available():
#                     action = None
#                 self._update_gamestuff(action=action)
#
#             self._update_gamescreen()
#             self._update_guistuff()
#             self._update_audio()
#             pygame.display.update()
#
#             if self.move_available():
#                 self.step_counter += 1
#
#                 ###############
#                 # Is it stuck?
#                 ###############
#                 if self.king.y == self.past_state_y:
#                     self.count_same_y += 1
#                 else:
#                     self.past_state_y = self.king.y
#                     self.count_same_y = 1
#
#                 is_stuck = True if self.count_same_y > 10 else False
#                 ###############
#
#                 right_state, left_state = self.compute_nearest()
#
#                 state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount] + \
#                         right_state + left_state
#
#                 ##################################################################################################
#                 # Define the reward from environment                                                             #
#                 ##################################################################################################
#                 s1 = [self.king.levels.current_level, self.king.y]
#                 reward = - self.Phi(s0, s1)
#
#                 # if is_stuck:
#                 # 	reward -= self.count_same_y
#                 ####################################################################################################
#
#                 done = True if self.step_counter > self.max_step else False
#                 return state, reward, done
#
#     def running(self):
#         """
#         play game with keyboard
#         :return:
#         """
#         self.reset()
#         while True:
#             right_state, left_state = self.compute_nearest()
#             print(right_state, left_state)
#             # state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
#             # print(state)
#             self.clock.tick(self.fps)
#             self._check_events()
#             if not os.environ["pause"]:
#                 self._update_gamestuff()
#
#             self._update_gamescreen()
#             self._update_guistuff()
#             self._update_audio()
#             pygame.display.update()
#
#     def _check_events(self):
#
#         for event in pygame.event.get():
#
#             if event.type == pygame.QUIT:
#                 self.environment.save()
#
#                 self.menus.save()
#
#                 sys.exit()
#
#             if event.type == pygame.KEYDOWN:
#
#                 self.menus.check_events(event)
#
#                 if event.key == pygame.K_c:
#
#                     if os.environ["mode"] == "creative":
#
#                         os.environ["mode"] = "normal"
#
#                     else:
#
#                         os.environ["mode"] = "creative"
#
#             if event.type == pygame.VIDEORESIZE:
#                 self._resize_screen(event.w, event.h)
#
#     def _update_gamestuff(self, action=None):
#
#         self.levels.update_levels(self.king, self.babe, agentCommand=action)
#
#     def _update_guistuff(self):
#
#         if self.menus.current_menu:
#             self.menus.update()
#
#         if not os.environ["gaming"]:
#             self.start.update()
#
#     def _update_gamescreen(self):
#
#         pygame.display.set_caption("Jump King At Home XD - :{} FPS".format(self.clock.get_fps()))
#
#         self.game_screen.fill(self.bg_color)
#
#         if os.environ["gaming"]:
#             self.levels.blit1()
#
#         if os.environ["active"]:
#             self.king.blitme()
#
#         if os.environ["gaming"]:
#             self.babe.blitme()
#
#         if os.environ["gaming"]:
#             self.levels.blit2()
#
#         if os.environ["gaming"]:
#             self._shake_screen()
#
#         if not os.environ["gaming"]:
#             self.start.blitme()
#
#         self.menus.blitme()
#
#         self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))
#
#     def _resize_screen(self, w, h):
#
#         self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SRCALPHA)
#
#     def _shake_screen(self):
#
#         try:
#
#             if self.levels.levels[self.levels.current_level].shake:
#
#                 if self.levels.shake_var <= 150:
#
#                     self.game_screen_x = 0
#
#                 elif self.levels.shake_var // 8 % 2 == 1:
#
#                     self.game_screen_x = -1
#
#                 elif self.levels.shake_var // 8 % 2 == 0:
#
#                     self.game_screen_x = 1
#
#             if self.levels.shake_var > 260:
#                 self.levels.shake_var = 0
#
#             self.levels.shake_var += 1
#
#         except Exception as e:
#
#             print("SHAKE ERROR: ", e)
#
#     def _update_audio(self):
#
#         for channel in range(pygame.mixer.get_num_channels()):
#
#             if not os.environ["music"]:
#
#                 if channel in range(0, 2):
#                     pygame.mixer.Channel(channel).set_volume(0)
#
#                     continue
#
#             if not os.environ["ambience"]:
#
#                 if channel in range(2, 7):
#                     pygame.mixer.Channel(channel).set_volume(0)
#
#                     continue
#
#             if not os.environ["sfx"]:
#
#                 if channel in range(7, 16):
#                     pygame.mixer.Channel(channel).set_volume(0)
#
#                     continue
#
#             # pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))
# FOR TEST
class JKGame:
    """ Overall class to manga game aspects """

    def __init__(self, max_step=float('inf'), cheating_level=0):

        self.cheating_level = cheating_level

        self.cheating_location = {0: (230, 298, "left"), 1: (330, 245, "right"), 2: (240, 245, "right"),
                                  3: (150, 245, "right")}

        pygame.init()

        self.environment = Environment()

        self.clock = pygame.time.Clock()

        self.fps = int(os.environ.get("fps"))

        self.bg_color = (0, 0, 0)

        self.screen = pygame.display.set_mode((
                                              int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")),
                                              int(os.environ.get("screen_height")) * int(
                                                  os.environ.get("window_scale"))),
                                              pygame.HWSURFACE | pygame.DOUBLEBUF)  # |pygame.SRCALPHA)

        self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)  # |pygame.SRCALPHA)

        self.game_screen_x = 0

        pygame.display.set_icon(pygame.image.load("images/sheets/JumpKingIcon.ico"))

        self.levels = Levels(self.game_screen)

        self.king = King(self.game_screen, self.levels)

        self.babe = Babe(self.game_screen, self.levels)

        self.menus = Menus(self.game_screen, self.levels, self.king)

        self.start = Start(self.game_screen, self.menus)

        self.step_counter = 0
        self.max_step = max_step

        self.visited = {}

        self.abs_total_height = 43 * 360 - 144

        # 43 semantic pictures
        self.semantic_ref_array = Create_ref_array()

        # New
        self.flag_stuck = 0
        self.nearest_platform_dist = 0
        self.nearest_platform_angle = 0

        pygame.display.set_caption('Jump King At Home XD')

    def reset(self):
        self.king.reset(self.cheating_location[self.cheating_level][0], self.cheating_location[self.cheating_level][1],
                        self.cheating_location[self.cheating_level][2])
        self.levels.reset(self.cheating_level)
        os.environ["start"] = "1"
        os.environ["gaming"] = "1"
        os.environ["pause"] = ""
        os.environ["active"] = "1"
        os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
        os.environ["session"] = "0"

        self.step_counter = 0
        done = False
        # Get init collision states
        (init_collision_states_array, init_collision_states_list) = Get_states_big(
            self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x, self.king.rect_y
        )
        state = [
                    self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount,  # 0
                ] + init_collision_states_list  # 4 + 132 states

        self.visited = {}
        self.visited[(self.king.levels.current_level, self.king.y)] = 1

        return done, state

    def move_available(self):
        available = not self.king.isFalling \
                    and not self.king.levels.ending \
                    and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
        return available

    # potential-based reward shaping
    def Phi(self, s0, s1):
        abs_old_y = (s0[0] + 1) * 360 - s0[2]
        abs_new_y = (s1[0] + 1) * 360 - s1[2]
        diff_height = abs_old_y - abs_new_y

        return diff_height

    def step(self, action):
        (collision_states_array, collision_states_list) = Get_states_big(
            self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x, self.king.rect_y
        )
        s0 = [
                 self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount,  # self.flag_stuck
             ] + collision_states_list
        old_level = self.king.levels.current_level
        old_x = self.king.x
        old_y = self.king.y

        # old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
        while True:
            self.clock.tick(500 * self.fps)
            self._check_events()
            if not os.environ["pause"]:
                if not self.move_available():
                    action = None
                self._update_gamestuff(action=action)

            self._update_gamescreen()
            self._update_guistuff()
            self._update_audio()
            pygame.display.update()

            if self.move_available():
                self.step_counter += 1
                # Judge if the agent is stuck   ????????????
                if self.king.levels.current_level == old_level \
                        and abs(old_x - self.king.x) <= 50 \
                        and old_y == self.king.y:
                    self.flag_stuck = 1
                else:
                    self.flag_stuck = 0
                # # Judge dist and angle
                # min_dist2 = 30000
                # for platform in self.levels.levels[self.levels.current_level].platforms:
                # 	if platform.y > self.king.y:
                # 		#print(self.nearest_platform_dist)
                # 		dist2 = (platform.x - self.king.x) ** 2 + (platform.y - self.king.y) ** 2
                # 		if dist2 < min_dist2 and platform.x != self.king.x:
                # 			min_dist2 = dist2
                # 			self.nearest_platform_dist = min_dist2 ** (1 / 2)
                # 			self.nearest_platform_angle = (platform.y - self.king.y) / (platform.x - self.king.x)

                # state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, self.flag_stuck,
                # 		 self.nearest_platform_dist, self.nearest_platform_angle]

                # Judge collision states
                (collision_states_array, collision_states_list) = Get_states_big(
                    self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x, self.king.rect_y
                )
                state = [
                            self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount,
                            # self.flag_stuck
                        ] + collision_states_list

                ##################################################################################################
                # Define the reward from environment                                                             #
                ##################################################################################################
                # Judge whether the king is stuck
                # if self.flag_stuck == 0:
                # 	if self.king.levels.current_level > old_level or (
                # 				self.king.levels.current_level == old_level and self.king.y < old_y):
                # 		reward = 0
                # 	# elif self.king.levels.current_level < old_level:
                # 	# reward = -1
                # 	else:
                # 		reward = -1
                # else:
                # 	reward = -10
                if self.king.levels.current_level > old_level or (
                                self.king.levels.current_level == old_level and self.king.y < old_y):
                    reward = 0
                # elif self.king.levels.current_level < old_level:
                # reward = -1
                else:
                    reward = -1

                s1 = state
                reward = reward - self.Phi(s0, s1)

                ####################################################################################################

                done = True if self.step_counter > self.max_step else False
                return s1, reward, done

    def running(self):
        """
        play game with keyboard
        :return:
        """
        self.reset()
        while True:
            # state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
            # print(state)
            self.clock.tick(self.fps)
            self._check_events()
            if not os.environ["pause"]:
                self._update_gamestuff()

            self._update_gamescreen()
            self._update_guistuff()
            self._update_audio()
            pygame.display.update()

            # Test when playing by players
            temp_players = Get_states(self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x,
                                      self.king.rect_y)

    def _check_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.environment.save()

                self.menus.save()

                sys.exit()

            if event.type == pygame.KEYDOWN:

                self.menus.check_events(event)

                if event.key == pygame.K_c:

                    if os.environ["mode"] == "creative":

                        os.environ["mode"] = "normal"

                    else:

                        os.environ["mode"] = "creative"

            if event.type == pygame.VIDEORESIZE:
                self._resize_screen(event.w, event.h)

    def _update_gamestuff(self, action=None):

        self.levels.update_levels(self.king, self.babe, agentCommand=action)

    def _update_guistuff(self):

        if self.menus.current_menu:
            self.menus.update()

        if not os.environ["gaming"]:
            self.start.update()

    def _update_gamescreen(self):

        pygame.display.set_caption("Jump King At Home XD - :{} FPS".format(self.clock.get_fps()))

        self.game_screen.fill(self.bg_color)

        if os.environ["gaming"]:
            self.levels.blit1()

        if os.environ["active"]:
            self.king.blitme()

        if os.environ["gaming"]:
            self.babe.blitme()

        if os.environ["gaming"]:
            self.levels.blit2()

        if os.environ["gaming"]:
            self._shake_screen()

        if not os.environ["gaming"]:
            self.start.blitme()

        self.menus.blitme()

        self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

    def _resize_screen(self, w, h):

        self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SRCALPHA)

    def _shake_screen(self):

        try:

            if self.levels.levels[self.levels.current_level].shake:

                if self.levels.shake_var <= 150:

                    self.game_screen_x = 0

                elif self.levels.shake_var // 8 % 2 == 1:

                    self.game_screen_x = -1

                elif self.levels.shake_var // 8 % 2 == 0:

                    self.game_screen_x = 1

            if self.levels.shake_var > 260:
                self.levels.shake_var = 0

            self.levels.shake_var += 1

        except Exception as e:

            print("SHAKE ERROR: ", e)

    def _update_audio(self):

        for channel in range(pygame.mixer.get_num_channels()):

            if not os.environ["music"]:

                if channel in range(0, 2):
                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            if not os.environ["ambience"]:

                if channel in range(2, 7):
                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            if not os.environ["sfx"]:

                if channel in range(7, 16):
                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))


def train(args):
    action_dict = {
        0: 'right',
        1: 'left',
        2: 'right+space',
        3: 'left+space',
        # 4: 'idle',
        # 5: 'space',
    }

    env = JKGame(max_step=5000, cheating_level=0)
    agent = TD3(state_size=args.state_size, action_size=args.action_size, gamma=args.gamma, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                batch_size = args.batch_size, buffer_capacity=args.buffer_capacity, tau=args.tau, random_seed=args.seed,
                policy_freq=args.policy_freq)
    time_start = time.time()        # Init start time
    ep_reward_list = []
    avg_reward_list = []
    best_avg_reward = -5000
    total_timesteps = 0

    epsilon = 1
    epsilon_decay = 0.9995
    epsilon_min = 0.01

    for ep in range(args.total_episodes):
        done, state = env.reset()
        episodic_reward = 0
        timestep = 0

        while not done:
            # Select action randomly or according to policy
            if np.random.uniform() > epsilon:
                action, action_logits = agent.policy(state)
            else:
                action = int(np.random.choice(args.action_size, 1))
                action_logits = torch.zeros(args.action_size)
                action_logits[action] = 1
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # Recieve state and reward from environment.
            next_state, reward, done = env.step(action)

            episodic_reward += reward
            sign = 1 if done else 0

            agent.add_to_replay_memory(state, action_logits.cpu(), reward, next_state, sign)

            state = next_state
            timestep += 1
            total_timesteps += 1

        ep_reward_list.append(episodic_reward)
        # Mean of last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)

        # Save agent info by the best avg_reward
        if ep%10 == 9 and avg_reward_list[-1] > best_avg_reward:
            best_avg_reward = avg_reward_list[-1]
            save_dict = {}
            save_dict["best_avg_reward"] = best_avg_reward
            save_dict["agent"] = agent
            torch.save(save_dict, "Weights/agent.pkl")


        s = (int)(time.time() - time_start)
        agent.learn_and_update_weights_by_replay(timestep)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Episode Reward: {:.2f}, Moving Avg.Reward: {:.2f}, Time: {:02}:{:02}:{:02}'
                .format(ep, total_timesteps, timestep,
                      episodic_reward, avg_reward, s//3600, s%3600//60, s%60))
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='td3')
    parser.add_argument('--state_size', type=int, default=4+132, metavar='N', help='dimension of state')
    parser.add_argument('--action_size', type=int, default=4, metavar='N', help='dimension of action')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discounted factor')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G', help='target smoothing coefficient(τ)')
    parser.add_argument('--lr-actor', type=float, default=0.0003, metavar='G', help='learning rate of actor')
    parser.add_argument('--lr-critic', type=float, default=0.0003, metavar='G', help='learning rate of critic')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='N', help='buffer_capacity')
    parser.add_argument('--max-steps', type=int, default=1600, metavar='N',
                        help='maximum number of steps of each episode')
    parser.add_argument('--total-episodes', type=int, default=3000, metavar='N', help='total training episodes')
    parser.add_argument('--policy-freq', type=int, default=2, metavar='N', help='update frequency of target network ')
    parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',
                        help='number of steps using random policy')
    args = parser.parse_args("")
    train(args)

    #Game = JKGame()
    #Game.running()