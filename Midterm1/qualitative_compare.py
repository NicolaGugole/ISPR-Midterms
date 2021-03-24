import cv2 
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from sift import *

# for reproducibility of resulting images --> python .\qualitative_compare.py 2 0 100
# ugly and repetitive code for qualitative comparison (shame on me)

# interestingly enough (found toying around) is that among the most 5 highest responsive kp for the image of the woman, 4 are about the eyes

imgNames = ["2_6_s.bmp","6_12_s.bmp"] # tree - woman
color=(255,0,0)
bar_color = ['b','g']

# ------------------------------------------------------------------------------------------------------------------ #

# plot (for the 2 images) keypoints in ordered position sys.argv[1] based on response (e.g. 0 takes the most responsive)
fig, ax = plt.subplots(nrows=2, ncols=2)
for i in range(len(imgNames)):
    img = cv2.imread(os.getcwd()+"\\chosen_imgs\\"+imgNames[i])
    sift = cv2.xfeatures2d.SIFT_create()
    # get grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect and display
    kp, des = sift.detectAndCompute(grayImg, None)
    sorted_kp_des = sorted(zip(kp,des), key=lambda x : x[0].response, reverse=True) # order based on response (Higher = Better)
    kp = [x for x,_ in sorted_kp_des]
    des = [x for _,x in sorted_kp_des]
    to_draw = int(sys.argv[1])
    cv2.drawKeypoints(grayImg,[kp[to_draw]],img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=color)
    ax[i][0].imshow(img)
    ax[i][1].bar(range(len(des[to_draw])),des[to_draw], color=bar_color[i])
# for fullscreen
figManager = plt.get_current_fig_manager() 
figManager.full_screen_toggle()
fig.suptitle(f"High response keypoint")
plt.show(block=False)

# ------------------------------------------------------------------------------------------------------------------ #

# plot (for the 2 images) keypoints in ordered position sys.argv[2] based on size (e.g. 0 takes the biggest)
fig, ax = plt.subplots(nrows=2, ncols=2)
for i in range(len(imgNames)):
    img = cv2.imread(os.getcwd()+"\\chosen_imgs\\"+imgNames[i])
    sift = cv2.xfeatures2d.SIFT_create()
    # get grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect
    kp, des = sift.detectAndCompute(grayImg, None)
    sorted_kp_des = sorted(zip(kp,des), key=lambda x : x[0].size, reverse=True) # order based on size (Higher = Better)
    kp = [x for x,_ in sorted_kp_des]
    des = [x for _,x in sorted_kp_des]
    to_draw = int(sys.argv[2])
    cv2.drawKeypoints(grayImg,[kp[to_draw]],img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=color)
    # preparing for plot
    ax[i][0].imshow(img)
    ax[i][1].bar(range(len(des[to_draw])),des[to_draw], color=bar_color[i])
# for fullscreen
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
fig.suptitle(f"Biggest size keypoint")
plt.show(block=False)

# ------------------------------------------------------------------------------------------------------------------ #

# plot (for the 2 images) keypoints in ordered position sys.argv[3] based on size again (e.g. 0 takes the biggest) 
# aiming at proving how keypoints alone are not enough to describe an image
fig, ax = plt.subplots(nrows=2, ncols=2)
for i in range(len(imgNames)):
    img = cv2.imread(os.getcwd()+"\\chosen_imgs\\"+imgNames[i])
    sift = cv2.xfeatures2d.SIFT_create()
    # get grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect and display
    kp, des = sift.detectAndCompute(grayImg, None)
    sorted_kp_des = sorted(zip(kp,des), key=lambda x : x[0].size, reverse=True)
    kp = [x for x,_ in sorted_kp_des]
    des = [x for _,x in sorted_kp_des]
    to_draw = int(sys.argv[3])
    cv2.drawKeypoints(grayImg,[kp[to_draw]],img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=color)
    # preparing for plot
    ax[i][0].imshow(img)
    ax[i][1].bar(range(len(des[to_draw])),des[to_draw], color=bar_color[i])
    # ax[i][1].set_title(to_draw)
# for fullscreen
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
fig.suptitle(f"Not so descriptive keypoint")
plt.show(block=True)
