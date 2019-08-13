import numpy as np
import cv2

MIN_MATCH_COUNT=5

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
searchParam=dict(checks = 100)
flann=cv2.FlannBasedMatcher(flannParam,searchParam)

img_check_A = cv2.imread("gray_a2.jpg",0)
trainKP_A,trainDesc_A=detector.detectAndCompute(img_check_A,None)

img_check_B = cv2.imread("gray_b2.jpg",0)
trainKP_B,trainDesc_B=detector.detectAndCompute(img_check_B,None)



cap = cv2.VideoCapture(0)

while True:
       ret, source=cap.read()
       cv2.rectangle(source, (300, 300), (100, 100), (0, 255, 0), 2)
       crop_image = source[100:300, 100:300]
       
       gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
       blur = cv2.GaussianBlur(gray, (15, 15), 0)
       
       corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
       corners = np.int0(corners)
       for corner in corners:
            x,y = corner.ravel()
            cv2.circle(crop_image,(x,y),3,255,-1)
            cv2.circle(gray,(x,y),3,255,-1)
    
       
       queryKP, queryDesc= detector.detectAndCompute(gray,None)
       matches_A=flann.knnMatch(queryDesc,trainDesc_A,k=2)
       matches_B=flann.knnMatch(queryDesc,trainDesc_B,k=2)

       goodMatch_A=[]
       goodMatch_B=[]

       for a1,a2 in matches_A:
        if(a1.distance<0.75*a2.distance):
            goodMatch_A.append(a1)
       for b1,b2 in matches_B:
        if(b1.distance<0.75*b2.distance):
            goodMatch_B.append(b1)


       if(len(goodMatch_A)>MIN_MATCH_COUNT):
           cv2.putText(source, "A", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

       else:
       	   if(len(goodMatch_B)>MIN_MATCH_COUNT):
                cv2.putText(source, "B", (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           else:
       	        print "Not Enough match found in A - %d/%d"%(len(goodMatch_A),MIN_MATCH_COUNT)
       	        print "Not Enough match found in B - %d/%d"%(len(goodMatch_B),MIN_MATCH_COUNT)

       cv2.imshow('Corner',source)
       cv2.imshow('gray',gray)
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()

