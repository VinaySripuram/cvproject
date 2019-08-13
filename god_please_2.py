import cv2 
import numpy as np
import math

MIN_MATCH_COUNT=5

detector=cv2.xfeatures2d.SIFT_create()


FLANN_INDEX_KDTREE = 0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})


img_check_A = cv2.imread("common.jpg",0)
trainKP,trainDesc_A=detector.detectAndCompute(img_check_A,None)



cap = cv2.VideoCapture(0)

while True:
    ret, source=cap.read()
    cv2.rectangle(source, (300, 300), (100, 100), (0, 255, 0), 2)
    crop_image = source[100:300, 100:300]
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, QueryImg = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    queryKP, queryDesc= detector.detectAndCompute(QueryImg,None)
    matches_A=flann.knnMatch(queryDesc,trainDesc_A,k=2)
    
    
    goodMatch_A=[]
    
    for m,n in matches_A:
        if(m.distance<0.75*n.distance):
            goodMatch_A.append(m)
    
        
    if(len(goodMatch_A)>MIN_MATCH_COUNT):
       cv2.putText(source, "A", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    else:
        cv2.putText(source, "not enough matches", (5, 50),
cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',source)
    cv2.imshow('threshold',QueryImg)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()