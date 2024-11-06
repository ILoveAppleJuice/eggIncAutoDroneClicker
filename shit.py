import cv2
from PIL import ImageGrab
import numpy as np
import pyautogui
import time
import math

template = cv2.imread('cross.png')
dim = template.shape

bbox=(600,100,1320,900) # may need to adjust this depending on screen size
targetColor = [184,140,68] # may also need to adjust this. This is the color of the box on the drone to look for

boundary = (targetColor,targetColor)

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
    #grab screen image
    img = ImageGrab.grab(bbox) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img = np.array(img) #this is the array obtained from conversion

    #filter out so only the color of the drone box
    # create NumPy arrays from the boundaries
    lower = np.array(boundary[0], dtype = "uint8")
    upper = np.array(boundary[1], dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)


    ex = False
    speed = (0,0)
    pos = None
    #use blob detection to find clumps
    keypoints = detector.detect(mask)

    #go through the blobs and check with previous list of blobs to see if any are moving
    point = None
    for p in keypoints:
        #this logic is so shitty
        change = True
        for p_ in prevKeypoints:
            #if there are previous blobs in the same position means that it is not moving -> not a drone
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
            #calculate the speed so can extrapolate the future position
            speed = (pos[0]-prevPos[0],pos[1]-prevPos[1])
        else:
            start = time.time() # this has no use
        prevPos = pos
        ...
    else:
        ...
        prevPos = None
    prevKeypoints = keypoints
    
    mask = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if pos and ex: #if a drone is found
        
        extrapolatedPos = (int(pos[0] + speed[0]),int(pos[1] + speed[1]))

        pyautogui.moveTo(bbox[0] + extrapolatedPos[0], bbox[1] + extrapolatedPos[1])
        pyautogui.leftClick()

        mask = cv2.circle(mask,(int(pos[0] + speed[0]),int(pos[1] + speed[1])),5,(0,255,0),2)
        mask = cv2.line(mask,(int(pos[0]),int(pos[1])),(int(pos[0] + speed[0]),int(pos[1] + speed[1])),(0,255,0),2)
    



    # looks for the X thing to close menus
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #confidence above 0.6
    if max_val > 0.6:
        closeMid = (max_loc[0]+int(dim[0]/2),max_loc[1]+int(dim[1]/2))
        px = img[closeMid[1]][closeMid[0]]
        
        #check for all white pixels
        if px[0] == 255 and px[1] == 255 and px[2] == 255:
            # check that the close button has stopped moving to avoid accidentally clicking other stuff
            if math.fabs(((max_loc[0] - prevClosePos[0]) + (max_loc[1]-prevClosePos[1]))/2) <= 10:
                cv2.rectangle(mask,max_loc,(max_loc[0]+dim[0],max_loc[1]+dim[1]),(255,0,0),2)
                pyautogui.moveTo(bbox[0]+closeMid[0], bbox[1]+closeMid[1])
                pyautogui.leftClick()
        
        prevClosePos = max_loc
        ...

    frame = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    
    cv2.imshow("test", img)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()