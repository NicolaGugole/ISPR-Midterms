import cv2
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy

# live object detection (objects given as parameters) using sift
# example of execution:  python .\live_sift.py img1 img2

# detect keypoints and descriptors
def applySiftAndGetKP(gray,original):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    cv2.drawKeypoints(gray,kp,original )
    return kp, des

# applied once to recognizable objects, avoiding useless overhead
def prepareImg(img, show_img = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = applySiftAndGetKP(gray,img)
    if show_img: 
        img_to_show = copy.deepcopy(img)
        cv2.drawKeypoints(gray,kp,img_to_show)
        cv2.imshow("",img_to_show)
        cv2.waitKey()
    return [gray, kp, desc]

# match frame with given object if at least MIN_MATCHES considerable kps are found
def matchWithImg(frame, img, MIN_MATCHES = 30):
    # unwrap
    grayImg = img[0]
    kpImg = img[1]
    descImg = img[2]

    # take gray frame
    grayFrame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply algo to frame 
    kpFrame, descFrame = applySiftAndGetKP(grayFrame,frame)

    # feature matching
    matcherType = 1
    if matcherType == 0:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    else: # FASTER
        flannParam=dict(algorithm=0,tree=5)
        matcher=cv2.FlannBasedMatcher(flannParam,{})


    matches = matcher.knnMatch(descFrame,descImg, k=2) # take 2 nearest matches for a single kp
    goodMatches = []
    minimumRate = 0.75
    for m,n in matches:
        if m.distance < minimumRate * n.distance: # to check if match is noisy or not
            goodMatches.append(m)
    if len(goodMatches)>=MIN_MATCHES: # good result, show matched img
        print("Found a match")
        resultImg = cv2.drawMatches(grayFrame, kpFrame, grayImg, kpImg, goodMatches, grayImg, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        print("Super sad, no match here, going on Tinder..")
        print(f"Found {len(goodMatches)}/{MIN_MATCHES}")
        resultImg = frame

    return resultImg

# toy function, tries to match keypoints found in left part of frame with ones on right
def matchLeftRight(frame):
    # take gray frame
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # split frame and gray version in half (left, right)
    splitFrame = np.hsplit(frame, 2)
    lFrame, rFrame = splitFrame[0], splitFrame[1]
    splitGray = np.hsplit(gray, 2) # cut the gray in half (columns wise)
    lGray, rGray = splitGray[0], splitGray[1]

    # apply algo to left and right
    kpLeft, descLeft = applySiftAndGetKP(lGray,lFrame)
    kpRight, descRight = applySiftAndGetKP(rGray,rFrame)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) # using bruteforce matcher (use flann for faster results)
    matches = bf.match(descLeft,descRight)
    matches = sorted(matches, key = lambda x:x.distance)

    # create resulting img drawing matches between left and right
    resultImg = cv2.drawMatches(lGray, kpLeft, rGray, kpRight, matches[:50], rGray, flags=2) 
    return resultImg

# ------------------ MAIN ------------------- #

# to access video frames
capture = cv2.VideoCapture(0)

# True(no object given) -- match left right    False -- match frame with image
if len(sys.argv) > 1:
    execType = True
    matchImgs = []
    for i in range(1,len(sys.argv)):
        matchImgs.append(prepareImg(cv2.imread(sys.argv[i])))
else:
    execType = False

# get frame and try to match
while True:
    allGood, frame = capture.read()
    if allGood:
        if execType: # object recognition
            matchedFrame = frame
            for matchImg in matchImgs: # try to match with every input img
                matchedFrame = matchWithImg(matchedFrame, matchImg)
        else:
            matchedFrame = matchLeftRight(frame)
        cv2.imshow("FoundMatch",matchedFrame)
    if cv2.waitKey(1) == 27: # ESC key
        break
capture.release()
cv2.destroyAllWindows()