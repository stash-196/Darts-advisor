import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from PIL_CV2 import cv2pil, pil2cv
import board_config_cb as cfgc
import importlib
importlib.reload(cfgc)


# Draw circles
def draw_regions(src, tx, ty, in_cfg):
    img = src.copy()
    center = (in_cfg.CENTER[0] + tx, in_cfg.CENTER[1] + ty)
    img = cv2.circle(img, center, in_cfg.BULL_IN_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.BULL_OUT_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.DOUBLE_IN_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.DOUBLE_OUT_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.TRIPPLE_IN_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.TRIPPLE_OUT_R, in_cfg.COLOR, in_cfg.THICKNESS)
    img = cv2.circle(img, center, in_cfg.BOAED_EDGE_R, in_cfg.COLOR, in_cfg.THICKNESS)

    # line_pts = in_cfg.LINE_ENDPOINTS['5-20_17-3']
    # img = cv2.line(img, line_pts[0], line_pts[1], in_cfg.COLOR, in_cfg.THICKNESS)

    # Draw Lines
    # for key, value in in_cfg.LINE_ENDPOINTS.items():
    #     img = cv2.line(img, value[0], value[1], in_cfg.COLOR, in_cfg.THICKNESS)
    degrees = np.linspace(0, 2*np.pi, 20, endpoint=False)
    # print(degrees)
    r = in_cfg.DOUBLE_OUT_R
    for theta in degrees:
        x_vec = -int(in_cfg.DOUBLE_OUT_R * np.cos(theta + in_cfg.THETA_ZERO))
        y_vec = -int(in_cfg.DOUBLE_OUT_R * np.sin(theta + in_cfg.THETA_ZERO))
        end_pt = (center[0] + x_vec, center[1] + y_vec)
        img = cv2.line(img, center, end_pt, in_cfg.COLOR, in_cfg.THICKNESS)
    return img

def getScoreDescription(src, cfg):
    board_description = np.empty_like(src[:, :, 0], dtype=str)
    print("creating Board Description", end=""),
    height, width, dim = src.shape
    for i in range(height):
        for j in range(width):
            board_description[i, j] = cfg.getScore(np.array([i, j]))
        print("/", end=""),
        if i%100 == 0: print("/", i)
    return board_description


# Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)


img = draw_regions(img, 0, 0, cfgc)

target = (600, 400)
img = cv2.circle(img, target, 5, (0,0,255), cfgc.THICKNESS)
score = cfgc.getScore(target)
print(score)

# BOARD_DESCRIPTION = getScoreDescription(img, cfgc)

# Write drawn image to disk.
OUT_FILENAME = "outputs/draw_board_cb.jpg"
print("Saving aligned image : ", OUT_FILENAME)
cv2.imwrite(OUT_FILENAME, img)

# cv2.imshow('dartboard', img)
# cv2.waitKey()
# cv2.destroyAllWindows()