import cv2 
import matplotlib.pyplot as plt
import os
import math

# toy code for quantitative comparison between 2 very different and 2 quite similar imgs

print_matches = False # if before the subplot you want to take a look at the matches
for iter in range(2):
    if iter == 0: # compare face1 and tree
        img1 = cv2.imread(os.getcwd()+'/quantitative_comparison/2_6_s.bmp')  
        img2 = cv2.imread(os.getcwd()+'/quantitative_comparison/6_12_s.bmp') 
    else:         # compare face1 and face2
        img1 = cv2.imread(os.getcwd()+'/quantitative_comparison/6_13_s.bmp')  
        img2 = cv2.imread(os.getcwd()+'/quantitative_comparison/6_12_s.bmp') 

    # take grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift on images
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    # feature matching using euclidean distance (flann works faster..)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    if print_matches:
        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, None, flags=2)
        plt.imshow(img3),plt.show()


    fig, ax = plt.subplots(nrows=2, ncols=2)
    to_print = matches[0] # print nearest keypoints between imgs
    img4 = cv2.drawKeypoints(img1,[keypoints_1[to_print.queryIdx]],img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,0,0))
    img5 = cv2.drawKeypoints(img2,[keypoints_2[to_print.trainIdx]],img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,0,0))
    ax[0][0].imshow(img4)
    #ax[0][0].set_title(to_print.queryIdx) # to see relative kp index in img1
    ax[0][1].bar(range(len(descriptors_1[to_print.queryIdx])),descriptors_1[to_print.queryIdx], color='r')
    ax[1][0].imshow(img5)
    #ax[1][0].set_title(to_print.trainIdx) # to see relative kp index in img2
    ax[1][1].bar(range(len(descriptors_2[to_print.trainIdx])),descriptors_2[to_print.trainIdx], color='b')
    fig.suptitle(f"Nearest Match\nEuclidean Distance: {round(to_print.distance,2)}")
    #fig.suptitle(f"Len descriptors_1: {len(descriptors_1)} Len descriptors_2: {len(descriptors_2)} Len matches: {len(matches)}")
    plt.show()