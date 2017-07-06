
import os.path
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage import img_as_ubyte

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from sklearn.datasets import fetch_mldata

def presekSaPravom(bbox, height, width):
    if (bbox[3]+4 < x1 or bbox[1] > x2):
        return False
    if(bbox[2]+4 < y2):
        return False

    if(jedPrave(bbox[3]+1, bbox[2]+1) <= 0): #Ukoliko je kraj regiona ispod linije racunati kao da je ceo pravouganik presao liniju
        return False

    if(jedPrave(bbox[1], bbox[0]) > 0 and jedPrave(bbox[3]+4, bbox[0]) > 0 and jedPrave(bbox[1], bbox[2]+4) > 0 and jedPrave(bbox[3]+4, bbox[2]+4)>0):
        return False
    if(jedPrave(bbox[1], bbox[0]) < 0 and jedPrave(bbox[3]+4, bbox[0]) < 0 and jedPrave(bbox[1], bbox[2]+4) < 0 and jedPrave(bbox[3]+4, bbox[2]+4)<0):
        return False
    return True


def jedPrave(x, y):
    a = x*(y2-y1)
    b = y*(x1-x2)
    c = (x2*y1-x1*y2)
    return a + b + c; 

def pripremaTrainData(data):
    for i in range(0, len(data)):
        th = cv2.inRange(data[i].reshape(28, 28), 150, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        regions = regionprops(label(closing))
        if(len(regions) <= 1):
            bbox = regions[0].bbox
        else:
            max_height = 0
            max_width = 0
            for region in regions:
                t_bbox = region.bbox

                t_width = region.bbox[3] - region.bbox[1]
                t_height = region.bbox[2] - region.bbox[0]
                if(t_height> max_height):
                    if(t_width > max_width ):
                        bbox = t_bbox
                        max_height = t_height
                        max_width = t_width

        x = 0
        img = np.zeros((28, 28))
        rangeRows = range(bbox[0], bbox[2])
        rangeCols = range(bbox[1], bbox[3])
       
        for row in rangeRows:
            y = 0
            for col in rangeCols:
                img[x, y] = number[row, col]
                y = y + 1
            x = x + 1
        data[i] = img.reshape(1, 784)

def getNumberImage(bbox, img):
    min_row = bbox[0]
    height = bbox[2] - min_row
    min_col = bbox[1]
    width = bbox[3] - min_col
    rangeX = range(0, height)
    rangeY = range(0, width)
    img_number = np.zeros((28, 28))
    for x in rangeX:
        for y in rangeY:
            img_number[x, y] = img[min_row+x-1, min_col+y-1]
    return img_number


def addNumber(lista, broj, width, height, region, bbox):

    for tup in lista:
        if(tup[0] == broj and tup[1] < bbox[1]+5  and tup[2] < bbox[0]+5 and tup[3] == width):
            lista.remove(tup)
            lista.append((broj, bbox[1], bbox[0], width))
            return False
    lista.append((broj, bbox[1], bbox[0], width))


mnist = fetch_mldata('MNIST original')
DIR = 'C:\\Users\Dule\\Desktop\\Projekat'
mnistFile = 'mnistPrepared'

if(os.path.exists(os.path.join(DIR, mnistFile)+'.npy') == False):
    train = mnist.data
    pripremaTrainData(train)
    np.save(os.path.join(DIR, mnistFile), train)
else:    
    train = np.load(os.path.join(DIR, mnistFile)+'.npy')
    
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, mnist.target)

DIR = DIR+'\\Videos'
close_kernel = np.ones((4, 4), np.uint8)
eros_kernel = np.ones((2, 2), np.uint8)

video_names = [os.path.join(DIR, name) for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
videoNamesRange = range(0, len(video_names))
for vid_num in videoNamesRange :
    frameNum = 0
    lista_brojeva=[]
    cap = cv2.VideoCapture(video_names[vid_num])
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frameNum%2 != 0):
            frameNum = frameNum + 1
            continue
        if(ret == False):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(frameNum < 1):
            erosion = cv2.erode(cv2.inRange(gray, 4, 55), eros_kernel, iterations=1)
            cv_skeleton = img_as_ubyte(skeletonize(erosion/255.0))
            lines = cv2.HoughLinesP(cv_skeleton, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            x1, y1, x2, y2 = lines[0][0]
        #th = cv2.inRange(gray, 163, 255)
        closing = cv2.morphologyEx(cv2.inRange(gray, 163, 255), cv2.MORPH_CLOSE, close_kernel)
        #gray_labeled = label(closing)
        regions = regionprops(label(closing))
        for region in regions:
            #bbox = region.bbox
            width = region.bbox[3]-region.bbox[1]
            height = region.bbox[2]-region.bbox[0]
            if(height <= 10 or presekSaPravom(region.bbox, height, width) == False):
                continue
            img_number = getNumberImage(region.bbox, gray)
            num = int(knn.predict(img_number.reshape(1, 784)))
            if(addNumber(lista_brojeva, num, width, height, region, region.bbox) == False):
                continue
            #print 'U frejmu '+ str(frameNum)+ '. prepoznat broj '+str(num)
        frameNum += 1
    suma=0
    brojac=0
    #while brojac < len(lista_brojeva) :
    #    suma += lista_brojeva[brojac][0]
    #    broj+=1
    for tup in lista_brojeva:
        suma += tup[0]
    
    print 'Video: ' + video_names[vid_num]
    print str(suma)+'\n'
cap.release()
cv2.destroyAllWindows()