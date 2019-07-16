#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import sys
import copy
import os
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


PATH = 'G:/computer vision/alcon2019/dataset/test/imgs/'
WRITEPATH = 'C:/Users/AlcanderLiu/Desktop/outputs'

if not os.path.exists(WRITEPATH):
    os.makedirs(WRITEPATH)

# In[3]:


def read_image(imgNum):
    # path = PATH+str(imgNum)+'.jpg'
    origImg = cv.imread(str(imgNum))
    
#     plt.imshow(origImg)
#     plt.title("origImg")
#     plt.show()
    
    return origImg


# In[4]:


# img = read_image(2231)


# In[5]:


'''
This function returns the ostu transform image of the input image with a Guassian blur with kernel of (11, 11)
'''
def otsu_img_blurred(img):
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grayImg, (11,11), 0)
    ret, otsuImg = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    return otsuImg


'''
This function returns the opening image of the input with a kernel of (11, 11)
'''
def open_img(img):
    kernel = np.ones((11,11), np.uint8)
    temp = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return temp

'''
This function returns the thinning image of the input
'''
def thin_img(img):
    temp = cv.ximgproc.thinning(img)
    return temp


'''
This fucntion returns all the stats for connencte components
Statistics output for each label, including the background label, see below for available statistics. Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
cv2.CC_STAT_WIDTH The horizontal size of the bounding box
cv2.CC_STAT_HEIGHT The vertical size of the bounding box
cv2.CC_STAT_AREA The total area (in pixels) of the connected component
'''
def find_cc(img):
    ccOutput = cv.connectedComponentsWithStats(img)
    return ccOutput


'''
This function returns the sum of every row in the input, divied by 255 to normalize
'''
def sum_row(img):
    return np.array([np.sum(row) for row in img])/255


'''
This function will crop the images base on the crop location into 3 separate images 
'''
def crop_img(imgNum, img, loc, method):
    title = str(imgNum)+'_Seg'
    loc = np.sort(loc)
    seg1 = img[:loc[0], :]
    seg2 = img[loc[0]:loc[1], :]
    seg3 = img[loc[1]:, :]
    segmentations = [seg1, seg2, seg3]
    
    for i in range(3):
        # plt.imshow(segmentations[i])
        cv.imwrite(os.path.join(WRITEPATH, title+str(i+1)+'_'+str(method)+'.jpg'), segmentations[i])
        
    return segmentations


# In[6]:


'''
First segmentation method
'''
def segmentation_1(imgNum, img):
    otsu = otsu_img_blurred(img)
    thinImg = thin_img(otsu)
    thinRowSum = sum_row(thinImg)
    
    charFlag = True
    cropLoc = np.array([], dtype = 'int')
    
    temp = 0
    
    # Find first non-zero
    for i in range(thinRowSum.shape[0]):
        if thinRowSum[i] != 0:
            temp = i
            break
    # print(temp)
    
    # Find the lower bound of each charater in the picture
    for row in range(temp, thinRowSum.shape[0]):
        if thinRowSum[row] == 0 and charFlag == True:
            cropLoc = np.append(cropLoc, row)
            charFlag = False            
        elif thinRowSum[row] != 0 and charFlag == False:
            charFlag = True
        if cropLoc.size == 2:
            break
    
    # print(cropLoc)
    if cropLoc.size != 2:
        raise Exception('cropLox size should be 2. The value of size is: {}'.format(cropLoc.size))
    else:
        return crop_img(imgNum, img, cropLoc, 1)
        
    


# In[7]:


'''
Second sementation method
'''
def segmentation_2(imgNum, img):
    cropLoc = np.array([])
    otsu = otsu_img_blurred(img)
    openImg = open_img(otsu)
    thinImg = thin_img(otsu)
    
    # ors := open row sum, trs := thin row sum
    ors = sum_row(openImg)
    trs = sum_row(thinImg)
    
    # otm:= one third mark of the image
    # ttm:= two third mark of the image
    otm = int(img.shape[0]*(1/3))
    ttm = int(img.shape[0]*(2/3))
    
    #
    break1NotDone = True 
    break2NotDone = True 
    conv = ors * trs
    i = 0
    while break1NotDone or break2NotDone:
        # If the one third mark is zero, find the first non zero above and blow it in the rowSum
        # If the one third mark isn't zero, find the first zero above and blow it in the rowSum
        if break1NotDone:
            if conv[otm] == 0:
                if conv[otm-i] != 0 or conv[otm+i] !=0:
                    cropLoc = np.append(cropLoc, (otm-i) if conv[otm-i] != 0 else (otm+i))
                    break1NotDone = False
                elif (otm+i) == ttm or (otm-i) == 0:
                    cropLoc = np.append(cropLoc, otm)
                    break1NotDone = False
            else:
                if conv[otm-i] == 0 or conv[otm+i] == 0:
                    cropLoc = np.append(cropLoc, (otm-i) if conv[otm-i] == 0 else (otm+i))
                    break1NotDone = False
                elif (otm+i) == ttm or (otm-i) == 0:
                    cropLoc = np.append(cropLoc, otm)
                    break1NotDone = False 
        

        # Same thing as the previous loop but at two third mark
        if break2NotDone:
            if conv[ttm] == 0:
                if conv[ttm-i] != 0 or conv[ttm+i] !=0:
                    cropLoc = np.append(cropLoc, (ttm-i) if conv[ttm-i] != 0 else (ttm+i))
                    break2NotDone = False
                elif (ttm+i) == img.shape[0] or (ttm-i) == otm:
                    cropLoc = np.append(cropLoc, ttm)
                    break2NotDone = False
            else:
                if conv[ttm-i] == 0 or conv[ttm+i] == 0:
                    cropLoc = np.append(cropLoc, (ttm-i) if conv[ttm-i] == 0 else (ttm+i))
                    break2NotDone = False
                elif (ttm+i) == img.shape[0] or (ttm-i) == otm:
                    cropLoc = np.append(cropLoc, ttm)
                    break2NotDone = False
        i += 1
        
    # print(cropLoc)
    if cropLoc.size != 2:
        raise Exception('cropLox size should be 2. The value of size is: {}'.format(cropLoc.size))
    else:
        return crop_img(imgNum, img, cropLoc.astype(int), 2)
    
    


# In[8]:


'''
Third segmentation method
'''

def segmentation_3(imgNum, img):
    cropLoc = np.array([])
    otsu = otsu_img_blurred(img)
    openImg = open_img(otsu)
    thinImg = thin_img(otsu)
    
    cc = find_cc(thinImg)
    # Get labels and totale number of labels
    labelNum = cc[0]
    labelImg = cc[1]

    # Get area for the each label
    lefts = cc[2][1:labelNum, 0]
    tops = cc[2][1:labelNum, 1]
    widths = cc[2][1:labelNum, 2]
    heights = cc[2][1:labelNum, 3]
    areas = cc[2][1:labelNum, 4]
    

    # Element-wise multiplication of heights and areas
    ha = heights * areas
    haSum = np.sum(ha)
    # print(ha)


    # If the ha value of the label is smaller than 15% of the haSum, it is unsifnificant and can be labeled by its overlapping counterpart
    sigLabel = np.array([], dtype = 'int')
    for i in range(labelNum - 1):
        if ha[i] >= 0.15*haSum:
    #       print(ha[i])
            sigLabel = np.append(sigLabel, i+1)
    
    # print(sigLabel.size)


    # Find each label's range, labelRange = [label, top, bottom]
    labelRange = np.array([], dtype = 'int')
    labelArea = np.array([], dtype = 'int')
    for i in range(labelNum - 1):
        labelRange = np.append(labelRange, [i+1, tops[i], tops[i]+heights[i]])
        labelArea = np.append(labelArea, [i+1, areas[i]])
    #     print(labelRange[3*i: 3*i+3])
    
#     print("\n")


    # Insignificant character overlapping
    # Check the overlapping, if it is insignificant, repalce with overlapping counterpart
    for i in range(labelNum - 1):
        for j in range(labelNum - 1):
            if i == j:
                continue
                
            if labelRange[3*i+1] >= labelRange[3*j+1] and labelRange[3*i+2] <= labelRange[3*j+2] and (labelRange[3*i] not in sigLabel):
                labelRange[3*i] = labelRange[3*j]
                labelArea[2*i] = labelRange[3*j]
                break
    # for i in range(labelNum-1):
    #     print(labelRange[3*i: 3*i+3])

        
    # Replace the labels on the image for the insignificant ones
    newLabelImg = copy.deepcopy(labelImg)
    for i in range(labelNum - 1):
        if (i+1) == labelRange[3*i]:
            continue
        else:
            newLabelImg = np.where(newLabelImg == (i+1), labelRange[3*i], newLabelImg)
            
    # If total labels are less than 2, the problem is out of the league of connected component
    if labelNum <= 3:
        return segmentation_10(imgNum, img)
    
    # If significant labels are 0, means the pictures are highly scattered
    elif sigLabel.size == 0 :
        return segmentation_10(imgNum, img)
    # If there is only one significant label
    elif sigLabel.size == 1:
        lab = sigLabel[0]
        height = heights[lab - 1]
        loc = labelRange[3*(lab-1)+1 :3*(lab-1)+3]
        
        if height <= img.shape[0] * 0.4 and loc[1] < img.shape[0] * (1/3):
            cropLoc = np.append(cropLoc, [loc[1], int(img.shape[0] * 2/3)])
            # print(1, cropLoc)
            return crop_img(imgNum, img, cropLoc.astype(int), 3)
        elif height <= img.shape[0] * 0.4 and loc[0] > img.shape[0] * (2/3):
            cropLoc = np.append(cropLoc, [int(img.shape[0] * 1/3), loc[0]])
            # print(2, cropLoc)
            return crop_img(imgNum,img, cropLoc.astype(int), 3)
        else:
            return segmentation_10(imgNum, img)
            
    # If there are exactly two significant labels, crop base on the labels' location
    elif sigLabel.size == 2 and labelNum > 3:
        if sigLabel[0] != 1:
            newLabelImg = np.where(newLabelImg == sigLabel[0], 1, newLabelImg)
    
        if sigLabel[1] != 2:
            newLabelImg = np.where(newLabelImg == sigLabel[1], 2, newLabelImg)
        
        for i in range(3, labelNum):
            newLabelImg = np.where(newLabelImg == int(i), int(3), newLabelImg)
           
        twonotdone = False
        threenotdone = False
        # print(newLabelImg.shape)
        for row in range(newLabelImg.shape[0]):
            cropFlag = False
            if 2 in newLabelImg[row] and cropFlag == False and twonotdone == False:
                cropLoc = np.append(cropLoc, row)
                cropFlag == True
                twonotdone = True
            if 3 in newLabelImg[row] and threenotdone == False:
                cropLoc = np.append(cropLoc, row)
                threenotdone = True
                break
        # print(3,cropLoc)
        return crop_img(imgNum, img, cropLoc.astype(int), 3)
        

    


# In[9]:


'''
Fifth segmentation method
Pure brute force
'''

def segmentation_5(imgNum, img):
    cropLoc = np.array([int(img.shape[0]*(1/3)), int(img.shape[0]*(2/3))])
    return crop_img(imgNum, img, cropLoc, 5)
    
    


# In[10]:


'''
Backup segmentation method
Pure brute force
'''

def segmentation_10(imgNum, img):
    cropLoc = np.array([int(img.shape[0]*(1/3)), int(img.shape[0]*(2/3))])
    # print('!!!seg 10')
    return crop_img(imgNum, img, cropLoc, 10)
    
    


# In[11]:


def main():
    img = cv.imread('G:/computer vision/alcon2019/dataset/test/imgs/14.jpg')
    print(img)
    #grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img)
    imgNum = 14
    '''
    result = segmentation_1(imgNum, img)
    if result is not None:     
        for img1 in result:
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            '''
    result = segmentation_2(imgNum, img)
    if result is not None:     
        for img1 in result:
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            '''
    result = segmentation_3(imgNum, img)
    if result is not None:     
        print(result)
        for img1 in result:
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            '''
    result = segmentation_5(imgNum, img)
    if result is not None:     
        for img1 in result:
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

segMethods = [segmentation_1, segmentation_2, segmentation_3, None, segmentation_5]
# In[ ]:

if __name__ == "__main__":
    main()




