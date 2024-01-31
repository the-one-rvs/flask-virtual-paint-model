import cv2 as cv
import HandTrackingModule as htm
import numpy as np


cap = cv.VideoCapture(1)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0

imgCanvas = np.zeros((480,640, 3),np.uint8)

def resize(img):
    resized_height = 350 #int(0.5 * (imgblank.shape[0]))
    resized_width = 110 #int(0.5 * (imgblank.shape[1]))


    img = cv.resize(img, (resized_width, resized_height), interpolation=cv.INTER_CUBIC)
    return img


imgblank = cv.imread('Resources/blank.png')
imgred = cv.imread('Resources/red.png')
imggreen = cv.imread('Resources/green.png')
imgblue = cv.imread('Resources/blue.png')
imgerase = cv.imread('Resources/erase.png')
imgdot = cv.imread('Resources/dot.png')
imgdisplay = cv.imread('Resources/Display.png')

resized_height = 50 #int(0.5 * (imgblank.shape[0]))
resized_width = 50 #int(0.5 * (imgblank.shape[1]))
imgdot = cv.resize(imgdot, (resized_width, resized_height), interpolation=cv.INTER_CUBIC)

resized_height = 480 #int(0.5 * (imgblank.shape[0]))
resized_width = 640*2 #int(0.5 * (imgblank.shape[1]))
imgdisplay = cv.resize(imgdot, (resized_width, resized_height), interpolation=cv.INTER_CUBIC)

imgX = imgblank

drawColor = (255,255,0)

while True:
    isTrue , img = cap.read()

    # imgBG = cv.imread('Resources/paint.png')

    cv.flip(img, 0)

    img = detector.findHands(img)
    lm = detector.findPosition(img, draw=False)
    lmlist = lm[0]

    img[50:100,0:50] = imgdot

    if len(lmlist)!=0:
        # xp,yp = 0,0
        # print(lmlist)

        #tip of index and middle finger 
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

    
        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            print ("Selection mode")
            if x1>530:
                if y1>0 and y1<87.5 :
                    imgX=imgred
                    drawColor = (0,0,255)
                if y1>87.5 and y1<175:
                    imgX=imgblue
                    drawColor = (255,0,0)
                if y1>175 and y1<262.5:
                    imgX=imggreen
                    drawColor = (0,255,0)
                if y1>262.5 and y1<=350:
                    imgX=imgerase
                    drawColor = (0,0,0)

            if x1>0 and x1<50 and y1>50 and y1<100:
                imgX = imgblank
                drawColor = (255,255,0)
            
            cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv.FILLED)

        if fingers[1] and fingers[2]==False:
            if drawColor != (255,255,0):
                cv.circle(img,(x1,y1),5,drawColor,cv.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp,yp = x1,y1 

                if drawColor ==  (0,0,0):
                    cv.line(img, (xp,yp), (x1,y1), drawColor , 50)
                    cv.line(imgCanvas, (xp,yp), (x1,y1), drawColor , 50)

                else:
                    cv.line(img, (xp,yp), (x1,y1), drawColor , 15)
                    cv.line(imgCanvas, (xp,yp), (x1,y1), drawColor , 15)

                xp,yp = x1,y1

    imgGray = cv. cvtColor (imgCanvas, cv.COLOR_BGR2GRAY)
    _,imgInv = cv. threshold(imgGray, 50, 255, cv. THRESH_BINARY_INV)
    imgInv = cv. cvtColor (imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and (img,imgInv)
    img = cv. bitwise_or (img,imgCanvas)


    imgX = resize(imgX)
    img[0:350,530:640]=imgX

    # img = cv.addWeighted(img,0.75,imgCanvas,0.25,0)
    imgdisplay[0:480,0:640] = img
    imgdisplay[0:480,640:640*2] = imgCanvas
    cv.imshow("img",img)
    cv.imshow("Canvas",imgdisplay)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()