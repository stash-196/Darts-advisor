import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from PIL_CV2 import cv2pil, pil2cv
import board_config as cfg
import importlib
importlib.reload(cfg)

# Read reference image
refFilename = "materials/shomen_cropped.jpg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Draw circles
img = cv2.circle(img, cfg.CENTER, cfg.BULL_IN_R, cfg.COLOR, cfg.THICKNESS)
img = cv2.circle(img, cfg.CENTER, cfg.BULL_OUT_R, cfg.COLOR, cfg.THICKNESS)
img = cv2.circle(img, cfg.CENTER, cfg.DOUBLE_IN_R, cfg.COLOR, cfg.THICKNESS)
img = cv2.circle(img, cfg.CENTER, cfg.DOUBLE_OUT_R, cfg.COLOR, cfg.THICKNESS)
img = cv2.circle(img, cfg.CENTER, cfg.TRIPPLE_IN_R, cfg.COLOR, cfg.THICKNESS)
img = cv2.circle(img, cfg.CENTER, cfg.TRIPPLE_OUT_R, cfg.COLOR, cfg.THICKNESS)


# line_pts = cfg.LINE_ENDPOINTS['5-20_17-3']
# img = cv2.line(img, line_pts[0], line_pts[1], cfg.COLOR, cfg.THICKNESS)

# Draw Lines
for key, value in cfg.LINE_ENDPOINTS.items():
    img = cv2.line(img, value[0], value[1], cfg.COLOR, cfg.THICKNESS)

# Write aligned image to disk.
OUT_FILENAME = "outputs/draw_board.jpg"
print("Saving aligned image : ", OUT_FILENAME)
cv2.imwrite(OUT_FILENAME, img)

cv2.imshow('dartboard', img)
cv2.waitKey()
cv2.destroyAllWindows()