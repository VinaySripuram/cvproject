import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 2)
    crop_image = frame[100:300, 100:300]
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255,
cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    (version, _, _) = cv2.__version__.split('.')

    if version is '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(),
cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version is '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),
cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

    drawing = np.zeros(crop_image.shape, np.uint8)
    max_area = 0

    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    centr = (cx, cy)
    cv2.circle(crop_image, centr, 5, [0, 0, 255], 2)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 2)

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
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


        # cv2.circle(crop_img,far,5,[0,0,255],-1)
    if count_defects == 1:
        cv2.putText(frame, "This is 2 ...", (5, 50),
cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    elif count_defects == 2:
        cv2.putText(frame, "This is 3 ...", (5, 50),
cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    elif count_defects == 3:
        cv2.putText(frame, "This is 4 ...", (50, 50),
cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    elif count_defects == 4:
        cv2.putText(frame, "HI!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    else:
        str = "This is a basic hand gesture recognizer"
        cv2.putText(frame, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

    cv2.imshow('threshold', thresh1)
    cv2.imshow('orginal', frame)
    cv2.imshow('drawing_2', gray)
    cv2.imshow('crop_image', crop_image)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()