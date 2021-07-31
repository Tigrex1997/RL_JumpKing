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

from skimage import draw,data
import matplotlib.pyplot as plt

from skimage import io,transform

if __name__ == "__main__":
    with open('Reward_hist/avg_reward_hist_list_2.data', 'rb') as filehandle:
        avg_reward_hist_list_2 = pickle.load(filehandle)
    with open('Reward_hist/avg_reward_hist_list_3.data', 'rb') as filehandle:
        avg_reward_hist_list_3 = pickle.load(filehandle)
    with open('Reward_hist/avg_reward_hist_list_3.data', 'rb') as filehandle:
        avg_reward_hist_list_3 = pickle.load(filehandle)

    avg_reward_hist_list_hang_sum = 0
    avg_reward_hist_list_hang = []
    reward_hist_list_hang = [
        -4999,-4999,-4999,-4999,-4999,-4845,-4934,-4919,-4600,-4513,
        -4665,-4830,-4853,-4853,-4851,-4495,-4561,-4578,-4373,-4561,
        -4829,-4853,-4486,-4501,-4372,-4400,-4495,-4502,-4495,-4847,
        -4496,-4696,-4831,-4354,-4831,-4839,-4189,-4349,-4316,-4652,
        -4432,-4345,-4183,-4541,-4642,-4329,-4663,-4637,-4667,-4644,
        -4637,-4703,-4707,-4704,-4707,-4707,-4707,-4707,-4707,-4852,
        -4707,-4707,-4706,-4707,-4707,-4706,-4707,-4501,-4706,-4707,
        -4701,-4703,-4686,-4853,-4695,-4702,-4691,-4687,-4703,-4700,
        -4694,-4583,-4363,-4359,-4952,-4459,-4967,-4566,-4366,-4661,
        -4577,-4473,-4682,-4693,-4497,-4486,-4400,-4687,-4497,-4488,
    ]
    for i in range(100):
        avg_reward_hist_list_hang_sum += reward_hist_list_hang[i]
        avg_reward_hist_list_hang.append(avg_reward_hist_list_hang_sum/(i+1))

    avg_reward_hist_list_35_sum = 0
    avg_reward_hist_list_35 = []
    reward_hist_list_35 = [
        -4999,-4999,-4999,-4999,-4999,-4844,-4845,-4576,-4573,-4676,
        -4677,-4481,-4560,-4497,-4455,-4565,-4553,-4552,-4367,-4564,
        -4658,-4808,-4569,-4339,-4558,-4547,-4453,-4381,-4565,-4372,
        -4363,-4559,-4387,-4370,-4673,-4372,-4673,-4387,-4370,-4565,
        -4547,-4381,-4565,-4372,-4673,-4455,-4560,-4677,-4576,-4676,
        -4481,-4560,-4497,-4455,-4559,-4387,-4370,-4673,-4372,-4553,
        -4547,-4453,-4381,-4565,-4372,-4677,-4481,-4560,-4497,-4455,
        -4547,-4381,-4565,-4372,-4673,-4381,-4565,-4372,-4363,-4559,
        -4387,-4547,-4547,-4381,-4565,-4372,-4547,-4381,-4565,-4370,
        -4844,-4673,-4339,-4558,-4547,-4453,-4381,-4565,-4372,-4363,
    ]
    for i in range(100):
        avg_reward_hist_list_35_sum += reward_hist_list_35[i]
        avg_reward_hist_list_35.append(avg_reward_hist_list_35_sum/(i+1))

    avg_reward_hist_mean = []
    for i in range(100):
        temp = avg_reward_hist_list_2[i] + avg_reward_hist_list_3[i] + avg_reward_hist_list_35[i] + avg_reward_hist_list_hang[i]
        avg_reward_hist_mean.append(temp/4)

    x_label = list(range(0, 100))

    plt.figure()
    #plt.plot(x_label, avg_reward_hist_list_2, 'red')
    #plt.plot(x_label, avg_reward_hist_list_3)
    # plt.plot(x_label, avg_reward_hist_list_hang)
    plt.plot(x_label, avg_reward_hist_list_35, 'red')
    plt.plot(x_label, avg_reward_hist_mean, 'blue', marker=',')



    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_2, avg_reward_hist_list_3,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill
    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_2, avg_reward_hist_list_35,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill
    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_2, avg_reward_hist_list_hang,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill
    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_3, avg_reward_hist_list_35,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill
    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_3, avg_reward_hist_list_hang,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill
    # Shade the area between y1 and line y=0
    plt.fill_between(x_label, avg_reward_hist_list_35, avg_reward_hist_list_hang,
                     facecolor="orange",  # The fill color
                     #color='blue',  # The outline color
                     alpha=0.05)  # Transparency of the fill


    plt.legend(['Best performance agent', 'Mean', 'Shaded region by trials'])

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Averaged Reward', fontsize=12)
    plt.xlim(0, 100)
    plt.title('Best average reward', fontsize=15)
    plt.grid()
    plt.savefig("Plotting/Avg_Reward.jpg")