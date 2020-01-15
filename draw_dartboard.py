import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from PIL_CV2 import cv2pil, pil2cv
import board_config_cb as cfgc
import importlib
import constraints
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

def getScoreDescription(src, in_cfg):
    height, width, dim = src.shape
    board_description_str = np.full((height, width), 'k', dtype=object)
    board_description_int = np.zeros((height, width))
    print("creating Board Description"),

    for h in range(height):
        for w in range(width):
            score = in_cfg.getScore(np.array([h, w]))
            board_description_str[h, w] = score
            if score in constraints.SCORES:
                board_description_int[h, w] = constraints.SCORES[score]
            if board_description_str[h, w] != score: print("um", board_description_str[h, w], "is not", score)
            # if board_description_str[h, w] is not ('' or "?"): print("!")
            # print(board_description_str[h, w])
        # print("/", end=""),
        if h%100 == 0: print("////", h)
    return board_description_str, board_description_int


# # Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)


img = draw_regions(img, 0, 0, cfgc)

# target = (600, 400)

center_of_regions = cfgc.getCenterOfRegions()
# print(center_of_regions)
# v = tuple(v)
for key, value in center_of_regions.items():
    # print(key, value)
    color = (0, 0, 255)
    if "_D" in key: color = (0, 255, 0)
    if "_T" in key: color = (0, 255, 255)
    img = cv2.circle(img, value, 5, color, 3)
# score = cfgc.getScore(target)


# # BOARD_DESCRIPTION = getScoreDescription(img, cfgc)

# # Write drawn image to disk.
OUT_FILENAME = "outputs/draw_board_and_centers_cb.jpg"
print("Saving aligned image : ", OUT_FILENAME)
cv2.imwrite(OUT_FILENAME, img)

# # cv2.imshow('dartboard', img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()