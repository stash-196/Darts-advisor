import numpy as np
import cv2

BLUR = (5,5)
BINARY_THRESHOLD_MIN = 30
BINARY_THRESHOLD_MAX = 255
CLAHE_CLIP_LIMIT = 5
CLAHE_TILE_SIZE = (10,10)

ARROW_BLUR = (5,5)
ARROW_BINARY_THRESHOLD_MIN = 50
ARROW_BINARY_THRESHOLD_MAX = 255
ARROW_CLAHE_CLIP_LIMIT = 20
ARROW_CLAHE_TILE_SIZE = (10,10)

def getDifference(gray1, gray2):
    # blur
    blur = BLUR
    # gray2 = cv2.blur(gray2, blur)
    # gray1 = cv2.blur(gray1, blur)

    #Gaussian blur
    gray2 = cv2.GaussianBlur(gray2, blur, 0)
    gray1 = cv2.GaussianBlur(gray1, blur, 0)

    #normalize
    # gray1 = cv2.equalizeHist(gray1)
    # gray2 = cv2.equalizeHist(gray2)

    #clahe
    # clahe = cv2.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE)
    # gray1 = clahe.apply(gray1)
    # gray2 = clahe.apply(gray2)

    #diff
    diff = cv2.subtract(gray2, gray1) + cv2.subtract(gray1, gray2)
    ret2, dif_thred = cv2.threshold(diff, BINARY_THRESHOLD_MIN, BINARY_THRESHOLD_MAX, cv2.THRESH_BINARY)
    # dif_thred = cv2.medianBlur(dif_thred, 6)
    return dif_thred, gray1, gray2, diff

if __name__ == '__main__':
    # Read background image
    bcgFilename = "materials/0104.1.jpg"
    print("Reading reference image : ", bcgFilename)
    img_bcg = cv2.imread(bcgFilename, cv2.IMREAD_COLOR)[:, 0:800]

    # Read new image
    newFilename = "materials/0104.2.jpg"
    print("Reading reference image : ", newFilename)
    img_new = cv2.imread(newFilename, cv2.IMREAD_COLOR)[:, 0:800]

    gray_bcg = cv2.cvtColor(img_bcg, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    diff_thred, gray1, gray2, diff = getDifference(gray_bcg, gray_new)

    # Write detected difference to disk.
    outFilename1 = "outputs/diff.jpg"
    outFilename2 = "outputs/diff_thred.jpg"
    print("Saving detected difference image : ", outFilename1, outFilename2)
    cv2.imwrite(outFilename1, diff)
    cv2.imwrite(outFilename2, diff_thred)
