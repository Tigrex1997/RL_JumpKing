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
import torchvision
import random
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt

import pickle

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


class NETWORK(torch.nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
		"""DQN Network example
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
		super(NETWORK, self).__init__()

		self.layer1 = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.layer2 = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.final = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.final(x)

		return x


class DDQN(object):
	def __init__(
			self
	):
		self.num_states = 4 + 132
		self.target_net = NETWORK(self.num_states, 4, 32).to(device)
		self.eval_net = NETWORK(self.num_states, 4, 32).to(device)

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

		self.memory_counter = 0
		self.memory_size = 50000
		self.memory = np.zeros((self.memory_size, self.num_states*2 + 3))

		self.epsilon = 1.0
		self.epsilon_decay = 0.9995
		self.alpha = 0.99

		self.batch_size = 64
		self.episode_counter = 0

		self.target_net.load_state_dict(self.eval_net.state_dict())

	def weights_load(self, weights_path):
		self.target_net.load_state_dict(torch.load(weights_path))
		self.eval_net.load_state_dict(torch.load(weights_path))
		#print('weights loaded successfully!')

	def memory_store(self, s0, a0, r, s1, sign):
		transition = np.concatenate((s0, [a0, r], s1, [sign]))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def select_action(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		if np.random.uniform() > self.epsilon:
			logit = self.eval_net(state.to(device))
			action = torch.argmax(logit, 1).item()
		else:
			action = int(np.random.choice(4, 1))

		return action

	def policy(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		logit = self.eval_net(state.to(device))
		action = torch.argmax(logit, 1).item()

		return action

	def train(self, s0, a0, r, s1, sign):
		if sign == 1:
			if self.episode_counter % 2 == 0:
				self.target_net.load_state_dict(self.eval_net.state_dict())
			self.episode_counter += 1

		self.memory_store(s0, a0, r, s1, sign)
		self.epsilon = np.clip(self.epsilon * self.epsilon_decay, a_min=0.01, a_max=None)

		# select batch sample
		if self.memory_counter > self.memory_size:
			batch_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[batch_index]
		batch_s0 = torch.tensor(batch_memory[:, :self.num_states]).float().to(device)
		batch_a0 = torch.tensor(batch_memory[:, self.num_states:self.num_states+1]).long().to(device)
		batch_r = torch.tensor(batch_memory[:, self.num_states+1:self.num_states+2]).float().to(device)
		batch_s1 = torch.tensor(batch_memory[:, self.num_states+2:2*self.num_states+2]).float().to(device)
		batch_sign = torch.tensor(batch_memory[:, 2*self.num_states+2:2*self.num_states+3]).long().to(device)

		q_eval = self.eval_net(batch_s0).gather(1, batch_a0)

		with torch.no_grad():
			maxAction = torch.argmax(self.eval_net(batch_s1), 1, keepdim=True)
			q_target = batch_r + (1 - batch_sign) * self.alpha * self.target_net(batch_s1).gather(1, maxAction)

		loss = self.criterion(q_eval, q_target)

		# backward
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf'), cheating_level=0):

		self.cheating_level = cheating_level

		self.cheating_location = {0:(230,298,"left"), 1:(330,245,"right"), 2:(240,245,"right"), 3:(150,245,"right")}

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

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

		self.abs_total_height = 43*360-144

		# 43 semantic pictures
		self.semantic_ref_array = Create_ref_array()

		# New
		self.flag_stuck = 0
		self.nearest_platform_dist = 0
		self.nearest_platform_angle = 0

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):
		self.king.reset(self.cheating_location[self.cheating_level][0],self.cheating_location[self.cheating_level][1],self.cheating_location[self.cheating_level][2])
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
	def Phi(self,s0,s1):
		abs_old_y = (s0[0]+1)*360 - s0[2]
		abs_new_y = (s1[0]+1)*360 - s1[2]
		diff_height = abs_old_y - abs_new_y

		return diff_height

	
	def step(self, action):
		(collision_states_array, collision_states_list) = Get_states_big(
			self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x, self.king.rect_y
		)
		s0 = [
			self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, #self.flag_stuck
		] + collision_states_list
		# print(len(s0))
		old_level = self.king.levels.current_level
		old_x = self.king.x
		old_y = self.king.y

		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		while True:
			self.clock.tick(500*self.fps)
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
						 self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, # self.flag_stuck
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
				return s1, reward, done, self.step_counter

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
			#state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			#print(state)
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				self._update_gamestuff()

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()

			# Test when playing by players
			(temp_players, temp_players2) = Get_states_big(
				self.semantic_ref_array, self.king.levels.current_level, self.king.rect_x, self.king.rect_y
			)

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

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

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


def train(flag_load, flag_not_record):
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		#4: 'idle',
		#5: 'space',
	}
	
	agent = DDQN()
	if flag_load == 1:
		weights_path = 'Weights/weights_episode35.pth'  # weights file path that is used to be loaded
		agent.weights_load(weights_path)
		print('Weights file loaded!')
	else:
		print('No weights to be loaded, train from scratch!')
	env = JKGame(max_step=5000, cheating_level=0)   # specify the starting level by cheating_level
	print(env.cheating_location)
	num_episode = 100
	avg_reward_sum = 0
	avg_reward_hist = []
	steps_hist = []

	for i in range(num_episode):
		done, state = env.reset()
		flag_fist_arrive = 0
		
		running_reward = 0
		while not done:
			action = agent.select_action(state)
			#print(action_dict[action])
			next_state, reward, done, temp_steps = env.step(action)

			if flag_fist_arrive == 0:
				if (next_state[0] == 1 and next_state[2] <= 180):
					steps_hist.append(temp_steps)
					print('Current steps: ', temp_steps)
					with open('Reward_hist/steps_hist.data', 'wb') as filehandle:
						pickle.dump(steps_hist, filehandle)
					flag_fist_arrive = 1


			# if reward < -2333:
			# 	break
			running_reward += reward
			sign = 1 if done else 0
			agent.train(state, action, reward, next_state, sign)
			state = next_state

		avg_reward_sum += running_reward
		avg_reward = avg_reward_sum/(i + 1)
		avg_reward_hist.append(avg_reward)
		with open('Reward_hist/avg_reward_hist_list.data', 'wb') as filehandle:
			pickle.dump(avg_reward_hist, filehandle)
		print ('episode: {}, reward: {}, avg_reward: {}'.format(i, running_reward, avg_reward))

		# save model parameters per episode
		if flag_not_record != 1:
			torch.save(agent.target_net.state_dict(), 'Weights/weights_episode%d.pth' % (agent.episode_counter - 1))
			print('weights of episode %d saved!' % (agent.episode_counter - 1))

	# Draw the avg reward hist
	# x = np.arange(num_episode)
	# plt.figure()
	# plt.plot(x, tr_his)
	# plt.plot(x, val_his)
	# plt.legend(['Training top1 accuracy', 'Validation top1 accuracy'])
	# plt.xticks(x)
	# plt.xlabel('Epoch')
	# plt.ylabel('Top1 Accuracy')
	# plt.title('MiniVGG')
	# plt.show()

			
if __name__ == "__main__":
	# ------------- Play the game manually -------------#
	# Game = JKGame()
	# Game.running()

	# ------------- Play the game by agent -------------#
	# Print the information
	print("PyTorch Version: ", torch.__version__)
	print("Torchvision Version: ", torchvision.__version__)
	# Detect if we have a GPU available
	if torch.cuda.is_available():
		print("Using the GPU!")
	else:
		print("WARNING: Could not find GPU! Using CPU only.")

	# Train  1. [flag_load == 0 -> no weights to load; Vice versa] 2. [flag_not_record == 0 -> record the weights]
	flag_load = 0
	flag_not_record = 0
	train(flag_load, flag_not_record)