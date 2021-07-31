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

if __name__ == "__main__":  # rect_x: 231  rect_y: 306
    file_path = 'Temp/Visual_state_window_for_paper.png'
    img = io.imread(file_path)

    new_img = transform.resize(img, (360, 480))

    rr, cc = draw.line(186, 51, 186, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])

    rr, cc = draw.line(354, 51, 354, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])

    rr, cc = draw.line(186+1*24, 51, 186+1*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(186+2*24, 51, 186+2*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(186+3*24, 51, 186+3*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(186+4*24, 51, 186+4*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(186+5*24, 51, 186+5*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])
    rr, cc = draw.line(186+6*24, 51, 186+6*24, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])


    rr, cc = draw.line(186, 51, 354, 51)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])

    rr, cc = draw.line(186, 431, 354, 431)
    draw.set_color(new_img, [rr, cc], [255, 0, 0])

    for i in range(18):
        rr, cc = draw.line(186, 51+(i+1)*20, 354, 51+(i+1)*20)
        draw.set_color(new_img, [rr, cc], [255, 0, 0])


    io.imshow(new_img)
    io.show()