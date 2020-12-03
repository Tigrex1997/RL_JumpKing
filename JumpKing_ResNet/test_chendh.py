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
import torchvision

import cv2
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Init resized size
    resized_h = 128
    resized_w = 128
    # Reshape and resize the semantic pictures of 43 levels
    folder_path = "./Semantic_MG"
    temp_tensor = torch.zeros(43, 3, resized_h, resized_w)
    output_tensor = torch.zeros(10, 3, resized_h, resized_w)
    temp_count = 0
    path_list = os.listdir(folder_path)
    path_list.sort(key=lambda x:int(x[:-4]))

    for filename in path_list:
        print(filename) #just for test
        # img is used to store the image data
        img = cv2.imread(folder_path + "/" + filename)
        # print("former:", img.shape)
        img_resized = cv2.resize(img, (resized_h, resized_w))
        matplotlib.image.imsave('Temp/test{}.png'.format(temp_count), img_resized)
        # print("after:", img_resized.shape)
        img_CHW = img_resized.transpose(2, 0, 1)  # (C, H, W)
        # print("after2:", img_CHW.shape)
        img_tensor = torch.tensor(img_CHW)
        # print("after3:", img_tensor.shape)
        temp_tensor[temp_count, :, :, :] = img_tensor[:, :, :]
        temp_count += 1