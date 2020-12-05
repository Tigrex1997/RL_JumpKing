#!/usr/bin/env python
#
#
#
#

import pygame
import collections
import os
from Platforms import Platforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cv2

if __name__ == "__main__":
    '''
    img1 = matplotlib.image.imread('MG/1.png')
    # np.array
    #print(img1.shape)  # (360, 480, 4)  (R, G. B, Alpha)
    #print(img1)
    # plt.imshow(img1)
    # plt.axis('off')
    # plt.show()

    img1_RGB = img1[:, :, 0:3]
    print(img1_RGB.shape)
    plt.imshow(img1_RGB)
    plt.axis('off')
    plt.show()

    matplotlib.image.imsave('Semantic_MG/1_semantic.png', img1_RGB)

    #img1_cv2 = cv2.imread('MG/1.png')
    #print(img1_cv2.shape)
    '''


    #----------------- Create semantic pictures for levels -----------------#
    # Init
    img1 = matplotlib.image.imread('MG/1.png')  # (360, 480, 4)
    img1_RGB = img1[:, :, 0:3]
    #print(init_Array.shape[0], init_Array.shape[1], init_Array.shape[2])
    semantic_h = img1_RGB.shape[0]
    semantic_w = img1_RGB.shape[1]
    semantic_c = img1_RGB.shape[2]

    pygame.init()
    shit = Platforms()

    # Loop
    for level in range(43):
        init_Array = np.zeros((360, 480, 3))  # (360, 480, 3)  0 -> black; 1 -> white

        platform = shit.platforms(level)

        '''
        print("self.levels[{}]\t=\t".format(level), end="")
        print('[', end="")
        print(*sorted(
            [(rect.x, rect.y, rect.width, rect.height, rect.slope, rect.slip, rect.support, rect.snow) for rect in
             platform], key=lambda l: type(l[4]) == tuple), sep=",\n\t\t\t\t\t", end="")
        print(']\n')
        '''

        for rect in platform:
            cur_rect_x = rect.x
            cur_rect_y = rect.y
            cur_rect_w = rect.width
            cur_rect_h = rect.height

            for h in range(semantic_h):
                for w in range(semantic_w):
                    if ((w >= cur_rect_x)
                        and (w < (cur_rect_x + cur_rect_w))
                        and (h >= cur_rect_y)
                        and (h < (cur_rect_y + cur_rect_h))):
                        init_Array[h, w, :] = [1, 1, 1]

        matplotlib.image.imsave('Semantic_MG/{}.png'.format(level + 1), init_Array)


    # # ----------------- Create semantic pictures for the king and babe poses -----------------#
    # img_base = matplotlib.image.imread('images/sheets/base.png')  # (288, 352, 4)
    # img_end_ani = matplotlib.image.imread('images/sheets/ending_animations.png')  # (384, 288, 4)
    # img_base_RGB = img_base[:, :, 0:3]
    # img_end_ani_RGB = img_end_ani[:, :, 0:3]
    # #print(img_base_RGB.shape)
    # #print(img_end_ani_RGB.shape)
    # # plt.imshow(img_base)
    # # plt.axis('off')
    # # plt.show()
    # # plt.imshow(img_base_RGB)
    # # plt.axis('off')
    # # plt.show()
    # base_h = img_base_RGB.shape[0]
    # base_w = img_base_RGB.shape[1]
    # end_ani_h = img_end_ani_RGB.shape[0]
    # end_ani_w = img_end_ani_RGB.shape[1]
    #
    # init_base = np.zeros(img_base_RGB.shape)
    # init_end_ani = np.zeros(img_end_ani_RGB.shape)
    #
    # # Process king poses
    # for h in range(base_h):
    #     for w in range(base_w):
    #         if (img_base_RGB[h, w, 0] != 0 or img_base_RGB[h, w, 1] != 0 or img_base_RGB[h, w, 2] != 0):
    #             init_base[h, w] = [2/100, 2/100, 2/100]  # Value [0, 0, 0] for validation, [2, 2, 2] for training
    #
    # # Process king and babe poses
    # for h in range(end_ani_h):
    #     for w in range(end_ani_w):
    #         if (img_end_ani_RGB[h, w, 0] != 0 or img_end_ani_RGB[h, w, 1] != 0 or img_end_ani_RGB[h, w, 2] != 0):
    #             init_end_ani[h, w] = [2/100, 2/100, 2/100]  # Value [0, 0, 0] for validation, [2, 2, 2] for training
    #
    # # Save pictures
    # matplotlib.image.imsave('Semantic_images/base_semantic.png', init_base)
    # matplotlib.image.imsave('Semantic_images/ending_animations_semantic.png', init_end_ani)