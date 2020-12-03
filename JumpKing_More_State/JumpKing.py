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
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
		self.target_net = NETWORK(7, 4, 32).to(device)
		self.eval_net = NETWORK(7, 4, 32).to(device)

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

		self.memory_counter = 0
		self.memory_size = 50000
		self.memory = np.zeros((self.memory_size, 17))

		self.epsilon = 1.0
		self.epsilon_decay = 0.9995
		self.alpha = 0.99

		self.batch_size = 64
		self.episode_counter = 0

		self.target_net.load_state_dict(self.eval_net.state_dict())

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
		batch_s0 = torch.tensor(batch_memory[:, :7]).float().to(device)
		batch_a0 = torch.tensor(batch_memory[:, 7:8]).long().to(device)
		batch_r = torch.tensor(batch_memory[:, 8:9]).float().to(device)
		batch_s1 = torch.tensor(batch_memory[:, 9:16]).float().to(device)
		batch_sign = torch.tensor(batch_memory[:, 16:17]).long().to(device)

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
		state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, 0, 0, 0]

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
		s0 = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, self.flag_stuck,
			  self.nearest_platform_dist, self.nearest_platform_angle]
		old_level = self.king.levels.current_level
		old_x = self.king.x
		old_y = self.king.y

		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		while True:
			self.clock.tick(200*self.fps)
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
				# Judge dist and angle
				min_dist2 = 30000
				for platform in self.levels.levels[self.levels.current_level].platforms:
					if platform.y > self.king.y:
						#print(self.nearest_platform_dist)
						dist2 = (platform.x - self.king.x) ** 2 + (platform.y - self.king.y) ** 2
						if dist2 < min_dist2 and platform.x != self.king.x:
							min_dist2 = dist2
							self.nearest_platform_dist = min_dist2 ** (1 / 2)
							self.nearest_platform_angle = (platform.y - self.king.y) / (platform.x - self.king.x)

				state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount, self.flag_stuck,
						 self.nearest_platform_dist, self.nearest_platform_angle]

				##################################################################################################
				# Define the reward from environment                                                             #
				##################################################################################################
				# Judge whether the king is stuck
				if self.flag_stuck == 0:
					if self.king.levels.current_level > old_level or (
								self.king.levels.current_level == old_level and self.king.y < old_y):
						reward = 0
					# elif self.king.levels.current_level < old_level:
					# reward = -1
					else:
						reward = -1
				else:
					reward = -10

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


def train():
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		#4: 'idle',
		#5: 'space',
	}
	
	agent = DDQN()
	env = JKGame(max_step=5000, cheating_level=0)   # specify the starting level by cheating_level
	print(env.cheating_location)
	num_episode = 100000

	for i in range(num_episode):
		done, state = env.reset()
		
		running_reward = 0
		while not done:
			action = agent.select_action(state)
			#print(action_dict[action])
			next_state, reward, done = env.step(action)

			# if reward < -2333:
			# 	break
			running_reward += reward
			sign = 1 if done else 0
			agent.train(state, action, reward, next_state, sign)
			state = next_state
		print ('episode: {}, reward: {}'.format(i, running_reward))

			
if __name__ == "__main__":
	# Game = JKGame()
	# Game.running()
	train()