from skimage.segmentation import clear_border
import imutils
import numpy as np
import cv2
img = cv2.imread('bmw.jpg')


# cv2.imshow('jay', img)

# cv2.waitKey(0)


def license(keep=5):
    img = cv2.imread('download.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    ######################################
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 25, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    ##############################################
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    ##############################################
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


##############################################
    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    cv2.imshow('jay', thresh)

    # thresh = cv2.dilate(thresh, None, iterations=2)
    # thresh = cv2.erode(thresh, None, iterations=1)
    cv2.waitKey(0)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
    return cnts


license()


def locate(clearBorder=False):
    candidates = license()
    img = cv2.imread('download.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lpCnt = None
    roi = None

    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w/float(h)

        if ar >= 1 and ar <= 5:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            if clearBorder:
                roi = clear_border(roi)
                print(roi)
                cv2.imshow('jay', licensePlate)
                cv2.waitKey(0)
                break

        return(roi, lpCnt)


print(locate())
