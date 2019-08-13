import cv2 
import numpy as np
import math
import time

MIN_MATCH_COUNT=10

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
searchParam=dict(checks = 50)
flann=cv2.FlannBasedMatcher(flannParam,searchParam)


img_check_A = cv2.imread("gray_a.jpg",0)
trainKP,trainDesc_A=detector.detectAndCompute(img_check_A,None)

img_check_B = cv2.imread("gray_b.jpg",0)
trainKP,trainDesc_B=detector.detectAndCompute(img_check_B,None)

img_check_C = cv2.imread("gray_c.jpg",0)
trainKP,trainDesc_C=detector.detectAndCompute(img_check_C,None)

img_check_D = cv2.imread("gray_d.jpg",0)
trainKP,trainDesc_D=detector.detectAndCompute(img_check_D,None)

img_check_E = cv2.imread("gray_e.jpg",0)
trainKP,trainDesc_E=detector.detectAndCompute(img_check_E,None)

img_check_F = cv2.imread("gray_f.jpg",0)
trainKP,trainDesc_F=detector.detectAndCompute(img_check_F,None)

img_check_G = cv2.imread("gray_g.jpg",0)
trainKP,trainDesc_G=detector.detectAndCompute(img_check_G,None)

img_check_H = cv2.imread("gray_h.jpg",0)
trainKP,trainDesc_H=detector.detectAndCompute(img_check_H,None)

img_check_I = cv2.imread("gray_i.jpg",0)
trainKP,trainDesc_I=detector.detectAndCompute(img_check_I,None)

img_check_J = cv2.imread("gray_j.jpg",0)
trainKP,trainDesc_J=detector.detectAndCompute(img_check_J,None)



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
       
   
    queryKP, queryDesc= detector.detectAndCompute(gray,None)
    matches_A=flann.knnMatch(queryDesc,trainDesc_A,k=2)
    matches_B=flann.knnMatch(queryDesc,trainDesc_B,k=2)
    matches_C=flann.knnMatch(queryDesc,trainDesc_C,k=2)
    matches_D=flann.knnMatch(queryDesc,trainDesc_D,k=2)
    matches_E=flann.knnMatch(queryDesc,trainDesc_E,k=2)
    matches_F=flann.knnMatch(queryDesc,trainDesc_F,k=2)
    matches_G=flann.knnMatch(queryDesc,trainDesc_G,k=2)
    matches_H=flann.knnMatch(queryDesc,trainDesc_H,k=2)
    matches_I=flann.knnMatch(queryDesc,trainDesc_I,k=2)
    matches_J=flann.knnMatch(queryDesc,trainDesc_J,k=2)
    
    goodMatch_A=[]
    goodMatch_B=[]
    goodMatch_C=[]
    goodMatch_D=[]
    goodMatch_E=[]
    goodMatch_F=[]
    goodMatch_G=[]
    goodMatch_H=[]
    goodMatch_I=[]
    goodMatch_J=[]



    for a1,a2 in matches_A:
        if(a1.distance<0.75*a2.distance):
            goodMatch_A.append(a1)
    for b1,b2 in matches_B:
        if(b1.distance<0.75*b2.distance):
            goodMatch_B.append(b1)
    for c1,c2 in matches_C:
        if(c1.distance<0.75*c2.distance):
            goodMatch_C.append(c1)
    for d1,d2 in matches_D:
        if(d1.distance<0.75*d2.distance):
            goodMatch_D.append(d1)
    for e1,e2 in matches_E:
        if(e1.distance<0.75*e2.distance):
            goodMatch_E.append(e1)
    for f1,f2 in matches_F:
        if(f1.distance<0.75*f2.distance):
            goodMatch_F.append(f1)
    for g1,g2 in matches_G:
        if(g1.distance<0.75*g2.distance):
            goodMatch_G.append(g1)
    for h1,h2 in matches_H:
        if(h1.distance<0.75*h2.distance):
            goodMatch_H.append(h1)
    for i1,i2 in matches_I:
        if(i1.distance<0.75*i2.distance):
            goodMatch_I.append(i1)
    for j1,j2 in matches_J:
        if(j1.distance<0.75*j2.distance):
            goodMatch_J.append(j1)




    if(len(goodMatch_A)>MIN_MATCH_COUNT):
       cv2.putText(source, "A", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    else:
       if(len(goodMatch_B)>MIN_MATCH_COUNT):
           cv2.putText(source, "B", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
       else:
           if(len(goodMatch_C)>MIN_MATCH_COUNT):
                cv2.putText(source, "C", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           else:
                if(len(goodMatch_D)>MIN_MATCH_COUNT):
                     cv2.putText(source, "D", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    if(len(goodMatch_E)>MIN_MATCH_COUNT):
                       cv2.putText(source, "E", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    else:
                        if(len(goodMatch_F)>MIN_MATCH_COUNT):
                            cv2.putText(source, "F", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        else:
                            if(len(goodMatch_G)>MIN_MATCH_COUNT):
                                 cv2.putText(source, "G", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            else:
                                if(len(goodMatch_H)>MIN_MATCH_COUNT):
                                      cv2.putText(source, "H", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                else:
                                    if(len(goodMatch_I)>MIN_MATCH_COUNT):
                                        cv2.putText(source, "I", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                    else:
                                        if(len(goodMatch_J)>MIN_MATCH_COUNT):
                                              cv2.putText(source, "J", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                        else:
                                              cv2.putText(source, "not enough matches", (2, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
                                              print "Not Enough match found in H - %d/%d"%(len(goodMatch_H),MIN_MATCH_COUNT)
                                              
    cv2.imshow('result',source)
    cv2.imshow('threshold',QueryImg)
    cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()