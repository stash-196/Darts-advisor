import numpy as np
import cv2

def cropOutObject(src):
     # 輪郭抽出
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours: 輪郭として抽出された点の座標の行列
    # cnt = contours[0]
    # cv.drawContours(img, [cnt], 0, (0,255,0), 3)
    max_area = 0
    middle = np.zeros(2)
    max_contour_left = 0
    max_contour_right = 0
    max_contour_top = 0
    max_contour_bottom = 0
    max_index = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 全輪郭の面積計算
        if max_area < area:
            max_area = area
            max_index = i

            max_contour_left = contours[i][0][0][0]
            max_contour_right = contours[i][0][0][0]
            max_contour_top = contours[i][0][0][1]
            max_contour_bottom = contours[i][0][0][1]
            for pt in contours[i]:
                # print(pt[0])
                if pt[0][0] < max_contour_left:
                    max_contour_left = pt[0][0]
                if pt[0][0] > max_contour_right:
                    max_contour_right = pt[0][0]
                if pt[0][1] < max_contour_top:
                    max_contour_top = pt[0][1]
                if pt[0][1] > max_contour_bottom:
                    max_contour_bottom = pt[0][1]

            print(max_contour_left)

    # calculate cropping region
    size = 100
    left = int(max_contour_left - size)
    right = int(max_contour_right + size)
    top = int(max_contour_top - size)
    bottom = int(max_contour_bottom + size)

    # crop src
    result = src[top:bottom, left:right]
    return result, left, right, top, bottom, contours, max_index

def getLinePts(src):
    print("HA!")
    rows, cols = src.shape
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_index = 0
    max_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 全輪郭の面積計算
        if max_area < area:
            max_area = area
            max_index = i
    if max_index > len(contours) - 1:
        print("error, maxindex:", max_index, "len(contours):", len(contours))
    cnt = contours[max_index]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    pt_lefty = (0, lefty)
    pt_righty = (cols-1, righty)
    return pt_lefty, pt_righty

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
    ret2, diff_thred = cv2.threshold(diff, BINARY_THRESHOLD_MIN, BINARY_THRESHOLD_MAX, cv2.THRESH_BINARY)
    print(diff_thred.dtype)
    print(diff_thred.shape)
    diff_thred = cv2.medianBlur(diff_thred, 7)
    return diff_thred, gray1, gray2, diff



if __name__ == '__main__':
    # Read background image
    bcgFilename = "materials/0111.0.jpg"
    print("Reading reference image : ", bcgFilename)
    img_bcg = cv2.imread(bcgFilename, cv2.IMREAD_COLOR)[:, 0:800]

    # Read new image
    newFilename = "materials/0111.2.jpg"
    print("Reading reference image : ", newFilename)
    img_new = cv2.imread(newFilename, cv2.IMREAD_COLOR)[:, 0:800]

    gray_bcg = cv2.cvtColor(img_bcg, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    diff_thred, gray1, gray2, diff = getDifference(gray_bcg, gray_new)

    # Write detected difference to disk.
    outFilename1 = "outputs/diff?.jpg"
    outFilename2 = "outputs/diff_thred?.jpg"
    print("Saving detected difference image : ", outFilename1, outFilename2)
    cv2.imwrite(outFilename1, diff)
    cv2.imwrite(outFilename2, diff_thred)
