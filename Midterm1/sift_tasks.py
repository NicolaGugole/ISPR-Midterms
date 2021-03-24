import cv2 
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import sys
from sift import *

# 3 possible executions given imgs in directory "chosen_imgs"
# - 0 will simply show sifted imgs with their histo descriptors overlapped
# - 1 will analyze imgs following argv[3] nearest matches comparing them with argv[2] indexed img in same directory
# - 2 will analyze imgs following argv[3] random matches comparing them with argv[2] indexed img in same directory

directory = 'chosen_imgs'
imgs = []
kps = []
dess = []
resultImgs = []
filenames = []
for filename in os.listdir(directory):
    filenames.append(filename)
    img = cv2.imread(os.getcwd()+"\\"+directory+"\\"+filename)
    imgs.append(img)
    kp,des,resultImg = siftImg(img)
    kps.append(kp)
    dess.append(des)
    resultImgs.append(resultImg)

for i in range(len(os.listdir(directory))):
    if int(sys.argv[1]) == 0: # show sifted img and descriptors
        showResult(resultImgs[i],dess[i], filenames[i])
    else:
        chosenOne = int(sys.argv[2])
        nMatches = int(sys.argv[3])
        if int(sys.argv[1]) == 1: # analyze with nearest matches
            getNMatches(dess[chosenOne],dess[i], filenames[chosenOne], filenames[i], minimumRate=0.60, nMatches=10, img1=imgs[chosenOne], img2=imgs[i],kp1=kps[chosenOne],kp2=kps[i])
        else: # analyze with random matches
            randMatches(dess[chosenOne],dess[i], filenames[chosenOne], filenames[i], nMatches=2, img1=imgs[chosenOne], img2=imgs[i],kp1=kps[chosenOne],kp2=kps[i])