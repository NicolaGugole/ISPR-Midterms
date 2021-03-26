import cv2
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy
import time

# UTILITY FUNCTIONS

def siftImg(img):
    sift = cv2.xfeatures2d.SIFT_create()
    #0 - get grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #1 − Detect and overlap to original img
    kp, des = sift.detectAndCompute(grayImg, None)
    resultImg = copy.deepcopy(img)
    cv2.drawKeypoints(grayImg,kp,resultImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, resultImg

# not used in the project but actually cool: can be used to plot all descriptors from 2 images side by side and get the avg distance between them
def showMatchedHist(matches,des1,des2,img1=None,kp1=None,img2=None,kp2=None,verbose=False):

    # showing only the descriptors histograms or also the matched images
    if img1 is None:
        nplots = 1
    else:
        nplots = 2
    fig, ax = plt.subplots(nrows=nplots, ncols=1)

    distCount = 0 # if verbose this will show distance of every single match

    # to plot multiple bars in an eye-pleasing manner
    xrange = range(len(des1[0]))
    offset = 0.4
    bardistance = 2
    xleft = [x*bardistance - offset for x in xrange]
    xright = [x*bardistance + offset  for x in xrange]

    # plot histo and eventually matches
    for match in matches:
        if verbose:
            print("Distance",distCount,":", match.distance)
        ax[0].bar(xleft, des1[match.queryIdx], color='b')
        ax[0].bar(xright, des2[match.trainIdx], color='r')
    ax[0].set_xticks([])
    if nplots > 1:
        result = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        ax[1].imshow(result)
    plt.show()

    print("Avg distance:",sum([match.distance for match in matches])/len(matches))

# not useful in the end, given 2 histos will compare following the 3 metrics - correlation/chisquared/intersection
def compareHists(matches,des1,des2):
    # to avoid modification of arrays
    nearest_des1 = []
    nearest_des2 = []
    for match in matches:
        nearest_des1.append(copy.deepcopy(des1[match.queryIdx]))
        nearest_des2.append(copy.deepcopy(des2[match.trainIdx]))

    compSelf = [] # ground truth
    comp12 = [] # comparing des1 with des2

    # useful for normalized intersection results
    nd1norm = []
    nd2norm = []
    for nd1, nd2 in zip(nearest_des1,nearest_des2):
        nd1norm.append(cv2.normalize(nd1,None,alpha=1,norm_type=cv2.NORM_L1))
        nd2norm.append(cv2.normalize(nd2,None,alpha=1,norm_type=cv2.NORM_L1))

    for method in range(3): # correlation - chisquared - intersection
        comparison1, comparison2 = 0,0
        if method == 2: #intersection
            nearest_des1 = nd1norm
            nearest_des2 = nd2norm
        for nd1, nd2 in zip(nearest_des1,nearest_des2):
            comparison1 += cv2.compareHist(nd1, nd1, method) 
            if method == 0:
                print(cv2.compareHist(nd1, nd2, method))
            comparison2 += cv2.compareHist(nd1, nd2, method) 
        comparison1, comparison2 = comparison1 / len(matches), comparison2 / len(matches) # take avg
        compSelf.append(comparison1)
        comp12.append(comparison2)
    
    # pretty printing    
    methods = ["Correlation ", "Chi-Square  ", "Intersection"]
    print('Method: ' + '\t\t' + '1 - 1' + '\t\t' + '1 - 2 ' + '\t\t')
    for c1, c2, m in zip(compSelf, comp12, methods):
        print(m + '\t\t' + str(round(c1,3)) + '\t\t' + str(round(c2,3)) + '\n\n') 

# plots result img with its descriptors as histo
def showResult(img,des,figname,block=True):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    des = np.random.permutation(des)[:10] # getting 10 random descriptors
    for d in des:
        plt.bar(range(len(d)),d)
    plt.show(block=block)

# analyzes imgs following the nMatches nearest matches 
def getNMatches(des1, des2, imgName1, imgName2, nMatches = 20, minimumRate = 0.75, img1=None, img2=None, kp1=None, kp2=None):
    # use flann for a faster matching operation
    flannParam=dict(algorithm=0,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    matches = flann.knnMatch(des1,des2, k=2)

    # find decent amount of matches trying to keep biggest possible distance between nearest and second nearest matching kp
    while True:
        goodMatches = []
        for m,n in matches:
            if m.distance < minimumRate * n.distance:
                goodMatches.append(m)
        if len(goodMatches)>=nMatches:
            break
        else:
            #print("Lowering threshold..")
            minimumRate += 0.001

    # get nMatches nearest keypoints
    goodMatches = sorted(goodMatches, key = lambda x:x.distance, reverse=False)[:nMatches]

    # plotting nearest nMatches descriptors
    print("Comparing",imgName1,"and",imgName2,"(final th =",minimumRate,")")
    showMatchedHist(goodMatches,des1,des2,img1,kp1,img2,kp2)

    # compare histogram results
    compareHists(goodMatches, des1, des2)


# analyzes imgs following nMatches random matches 
def randMatches(des1, des2, imgName1, imgName2, nMatches = 20, img1=None, img2=None, kp1=None, kp2=None):
    # use flann for a faster matching operation
    flannParam=dict(algorithm=0,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    matches = flann.knnMatch(des1,des2, k=2)

    randMatches = []
    for m,n in matches:
        randMatches.append(m)
    randMatches = np.random.permutation(randMatches)[:nMatches]

    # plotting nearest nMatches descriptors
    print("Comparing",imgName1,"and",imgName2)
    showMatchedHist(randMatches,des1,des2,img1,kp1,img2,kp2)

    # compare histogram results
    compareHists(randMatches, des1, des2)