import cv2
from PIL import ImageGrab
import numpy as np
import pyautogui
import time
import math

template = cv2.imread('cross.png')
dim = template.shape

bbox=(500,100,1420,900)


targetColor = [184,140,68]

boundary = ([193,152,79], [193,152,79])

params = cv2.SimpleBlobDetector.Params()

params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 1
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector.create(params)

prevClosePos = (0,0)

prevPos = None
start = 0

prevKeypoints = []
while True:
    img = ImageGrab.grab(bbox) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img = np.array(img) #this is the array obtained from conversion

    mask = None
    # create NumPy arrays from the boundaries
    lower = np.array(boundary[0], dtype = "uint8")
    upper = np.array(boundary[1], dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)

    ex = False
    speed = (0,0)
    pos = None
    keypoints = detector.detect(mask)


    point = None

    for p in keypoints:
        change = True
        for p_ in prevKeypoints:
            if math.fabs(((p.pt[0] - p_.pt[0]) + (p.pt[1]-p_.pt[1]))/2) <= 1:
                change = False
                break
        if change:
            point = p
            break

    if point:
        pos = point.pt
        if prevPos:
            ex = True
            speed = (pos[0]-prevPos[0],pos[1]-prevPos[1])
        else:
            start = time.time()
        prevPos = pos
        ...
    else:
        ...
        prevPos = None
    prevKeypoints = keypoints
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    mask = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if pos and ex:
        #print(time.time()-start)
        
        extrapolatedPos = (int(pos[0] + speed[0]),int(pos[1] + speed[1]))

        pyautogui.moveTo(bbox[0] + extrapolatedPos[0], bbox[1] + extrapolatedPos[1])
        pyautogui.leftClick()

        mask = cv2.circle(mask,(int(pos[0] + speed[0]),int(pos[1] + speed[1])),5,(0,255,0),2)
        mask = cv2.line(mask,(int(pos[0]),int(pos[1])),(int(pos[0] + speed[0]),int(pos[1] + speed[1])),(0,255,0),2)
    
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    

    #print(max_loc[1])
    #cv2.rectangle(img,max_loc,(max_loc[0]+dim[0],max_loc[1]+dim[1]),(255,0,0),2)
    #max_loc[1] < 160 and 
    if max_val > 0.6:
        closeMid = (max_loc[0]+int(dim[0]/2),max_loc[1]+int(dim[1]/2))
        px = img[closeMid[1]][closeMid[0]]
        
        if px[0] == 255 and px[1] == 255 and px[2] == 255:
            #print(((max_loc[0] - prevClosePos[0]) + (max_loc[1]-prevClosePos[1]))/2)
            if math.fabs(((max_loc[0] - prevClosePos[0]) + (max_loc[1]-prevClosePos[1]))/2) <= 10:
                cv2.rectangle(mask,max_loc,(max_loc[0]+dim[0],max_loc[1]+dim[1]),(255,0,0),2)
                pyautogui.moveTo(bbox[0]+closeMid[0], bbox[1]+closeMid[1])
                pyautogui.leftClick()
        
        prevClosePos = max_loc
        ...

    frame = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    
    cv2.imshow("test", mask)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()