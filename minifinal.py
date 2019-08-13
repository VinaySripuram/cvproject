import cv2 
import numpy as np
import math
import time
import subprocess

MIN_MATCH_COUNT=5

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=5)

searchParam=dict(checks = 100)
flann=cv2.FlannBasedMatcher(flannParam,searchParam)


img_check_A = cv2.imread("gray_a.jpg",0)
trainKP_A,trainDesc_A=detector.detectAndCompute(img_check_A,None)

img_check_B = cv2.imread("gray_b.jpg",0)
trainKP_B,trainDesc_B=detector.detectAndCompute(img_check_B,None)

img_check_C = cv2.imread("gray_c.jpg",0)
trainKP_C,trainDesc_C=detector.detectAndCompute(img_check_C,None)

img_check_D = cv2.imread("gray_d.jpg",0)
trainKP,trainDesc_D=detector.detectAndCompute(img_check_D,None)

img_check_E = cv2.imread("gray_e1.jpg",0)
trainKP,trainDesc_E=detector.detectAndCompute(img_check_E,None)

img_check_F = cv2.imread("gray_f.jpg",0)
trainKP,trainDesc_F=detector.detectAndCompute(img_check_F,None)

img_check_love = cv2.imread("love.jpg",0)
trainKP,trainDesc_love=detector.detectAndCompute(img_check_love,None)

img_check_gjob = cv2.imread("gjob.jpg",0)
trainKP,trainDesc_gjob=detector.detectAndCompute(img_check_gjob,None)

cap = cv2.VideoCapture(0)

while True:
    ret, source=cap.read()
    cv2.rectangle(source, (300, 300), (100, 100), (0, 255, 0), 2)
    

    crop_image = source[100:300, 100:300]
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, QueryImg = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (version, _, _) = cv2.__version__.split('.')

    if version is '3':
        image, contours, hierarchy = cv2.findContours(QueryImg.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version is '2':
        contours, hierarchy = cv2.findContours(QueryImg.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    gray1 = np.float32(gray)
    
    corners = cv2.goodFeaturesToTrack(gray1, 100, 0.01, 10)
    corners = np.int0(corners)
    
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(crop_image,(x,y),3,255,-1)


    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
    
    

    drawing_a = np.zeros(crop_image.shape, np.uint8)
    
    max_area = 0

    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    
    centr = (cx, cy)
    cv2.circle(crop_image, centr, 5, [0, 0, 255], 2)
    cv2.drawContours(drawing_a, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing_a, [hull], 0, (0, 255, 0), 2)

    

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(QueryImg, contours, -1, (0, 255, 0), 3)
   

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57


        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_image, far, 3, [0, 0, 255], -1)
        # dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_image, start, end, [0, 255, 0], 2)
    

    a=2
    if (a==1):
                     # cv2.circle(crop_img,far,5,[0,0,255],-1)
        if count_defects == 1:
          cv2.putText(source, "This is 2 ...", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2,[0,0,255], 3)
        elif count_defects == 2:
          cv2.putText(source, "This is 3 ...", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,255], 3)
        elif count_defects == 3:
          cv2.putText(source, "This is 4 ...", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,255], 3)
        elif count_defects == 4:
          cv2.putText(source, "This is 5 ...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,255], 3)
        else:
          str = "This is a basic hand gesture recognizer"
          cv2.putText(source, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        cv2.imshow('frame',source)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    
    if (a==2):
     

     queryKP, queryDesc= detector.detectAndCompute(gray,None)
     matches_A=flann.knnMatch(queryDesc,trainDesc_A,k=2)
     matches_B=flann.knnMatch(queryDesc,trainDesc_B,k=2)
     matches_C=flann.knnMatch(queryDesc,trainDesc_C,k=2)
     matches_D=flann.knnMatch(queryDesc,trainDesc_D,k=2)
     matches_E=flann.knnMatch(queryDesc,trainDesc_E,k=2)
     matches_F=flann.knnMatch(queryDesc,trainDesc_F,k=2)
    
     goodMatch_A=[]
     goodMatch_B=[]
     goodMatch_C=[]
     goodMatch_D=[]
     goodMatch_E=[]
     goodMatch_F=[]
     


     for a1,a2 in matches_A:
        if(a1.distance<0.4*a2.distance):
            goodMatch_A.append(a1)
     for b1,b2 in matches_B:
        if(b1.distance<0.4*b2.distance):
            goodMatch_B.append(b1)
     for c1,c2 in matches_C:
        if(c1.distance<0.4*c2.distance):
            goodMatch_C.append(c1)
     for d1,d2 in matches_D:
        if(d1.distance<0.4*d2.distance):
            goodMatch_D.append(d1)
     for e1,e2 in matches_E:
        if(e1.distance<0.4*e2.distance):
            goodMatch_E.append(e1)
     for f1,f2 in matches_F:
        if(f1.distance<0.4*f2.distance):
            goodMatch_F.append(f1)
    

     print ("Number of Good Matches in alphabetical order  - %d %d %d %d %d %d"%(len(goodMatch_A),len(goodMatch_B),len(goodMatch_C),len(goodMatch_D),len(goodMatch_E),len(goodMatch_F)))


     if(len(goodMatch_A)>len(goodMatch_B) and len(goodMatch_A)>len(goodMatch_C) and len(goodMatch_A)>len(goodMatch_D) and len(goodMatch_A)>len(goodMatch_E) and len(goodMatch_A)>len(goodMatch_F)):
       cv2.putText(source, "A", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
     else:
       if(len(goodMatch_B)>len(goodMatch_A) and len(goodMatch_B)>len(goodMatch_C) and len(goodMatch_B)>len(goodMatch_D) and len(goodMatch_B)>len(goodMatch_E) and len(goodMatch_B)>len(goodMatch_F)):
           cv2.putText(source, "B", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
       else:
           if(len(goodMatch_C)>len(goodMatch_A) and len(goodMatch_C)>len(goodMatch_B) and len(goodMatch_C)>len(goodMatch_D) and len(goodMatch_C)>len(goodMatch_E) and len(goodMatch_C)>len(goodMatch_F)):
                cv2.putText(source, "C", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           else:
            if(len(goodMatch_D)>len(goodMatch_A) and len(goodMatch_D)>len(goodMatch_B) and len(goodMatch_D)>len(goodMatch_C) and len(goodMatch_D)>len(goodMatch_E) and len(goodMatch_D)>len(goodMatch_F)):
                     cv2.putText(source, "D", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                if(len(goodMatch_E)>len(goodMatch_A) and len(goodMatch_E)>len(goodMatch_B) and len(goodMatch_E)>len(goodMatch_C) and len(goodMatch_E)>len(goodMatch_D) and len(goodMatch_E)>len(goodMatch_F)):
                       cv2.putText(source, "E", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    if(len(goodMatch_F)>len(goodMatch_A) and len(goodMatch_F)>len(goodMatch_B) and len(goodMatch_F)>len(goodMatch_C) and len(goodMatch_F)>len(goodMatch_D) and len(goodMatch_F)>len(goodMatch_E)):
                            cv2.putText(source, "F", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    else:
                            cv2.putText(source, "not enough matches", (3, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                       
                                                                
     cv2.imshow('result',source)
    # cv2.imshow('threshold',QueryImg)
     #cv2.imshow('gray',gray)

     if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if(a==3):
         queryKP, queryDesc= detector.detectAndCompute(gray,None)

         matches_love=flann.knnMatch(queryDesc,trainDesc_love,k=2)
         matches_gjob=flann.knnMatch(queryDesc,trainDesc_gjob,k=2)

         goodMatch_L=[]
         goodMatch_GJ=[]

         for l1,l2 in matches_love:
            if(l1.distance<0.4*l2.distance):
              goodMatch_L.append(l1)
         for gj1,gj2 in matches_gjob:
            if(gj1.distance<0.4*gj2.distance):
              goodMatch_GJ.append(gj1)

         #print "Number of Good Matches in love and good job  - %d %d "%(len(goodMatch_L),len(goodMatch_GJ))
         
         if ((count_defects == 1) and len(goodMatch_GJ)<= len(goodMatch_L)):
            cv2.putText(source, "VICTORY!!", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
         else:
            if(count_defects==4):
                cv2.putText(source, "HI / HELLO ", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                if(len(goodMatch_L)>len(goodMatch_GJ)):
                   cv2.putText(source, "  I LOVE YOU  ", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    if(len(goodMatch_GJ)>len(goodMatch_L)):
                      cv2.putText(source, "  GOOD JOB / KEEP IT UP  ", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
             
         cv2.imshow('frame',source)
         cv2.imshow('gray',gray)
          
         if cv2.waitKey(1) & 0xFF == ord('q'):
           break


cap.release()
cv2.destroyAllWindows()