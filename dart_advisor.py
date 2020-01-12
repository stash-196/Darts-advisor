import cv2
import numpy as np
from numpy.linalg import inv
import constraints
import detect_diff
import draw_dartboard
import board_config as cfg
import alignImages

import importlib
importlib.reload(constraints)
importlib.reload(detect_diff)
importlib.reload(draw_dartboard)
importlib.reload(cfg)
importlib.reload(alignImages)

from constraints import SCORES
from detect_diff import getDifference, getLine, cropOutObject
from draw_dartboard import draw_regions
from alignImages import alignImages


import sys
# print(sys.path)


# Read reference image
refFilename = "resources/masked_shomen.jpg"
print("Reading reference image : ", refFilename)
im_ref = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read background image
bcgFilename = "resources/0111.0.jpg"
print("Reading reference image : ", bcgFilename)
im_bcg = cv2.imread(bcgFilename, cv2.IMREAD_COLOR)

# Read mask
maskFilename = "resources/mask_shomen_regions.jpg"
print("Reading mask image : ", maskFilename)
mask_ref = cv2.imread(maskFilename, cv2.IMREAD_COLOR)
mask_ref = mask_ref[:, :, 0]

print(mask_ref.shape)

print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
im_bcg2ref, h_bcg2ref = alignImages(im_bcg, im_ref)
h_ref2bcg = np.linalg.inv(h_bcg2ref)
print("... homography attained")

height, width, channels = im_bcg.shape
mask_bcg = cv2.warpPerspective(mask_ref, h_ref2bcg, (width, height))

# mask background
im_bcg_masked = np.empty_like(im_bcg, dtype=np.uint8) #im_bcg.copy()
im_bcg_masked[:, :, 0] = cv2.bitwise_and(im_bcg[:, :, 0], mask_bcg)
im_bcg_masked[:, :, 1] = cv2.bitwise_and(im_bcg[:, :, 1], mask_bcg)
im_bcg_masked[:, :, 2] = cv2.bitwise_and(im_bcg[:, :, 2], mask_bcg)

# Read dart
thrownFilename = "resources/0111.2.jpg"
print("Reading mask image : ", thrownFilename)
im_thrown = cv2.imread(thrownFilename, cv2.IMREAD_COLOR)
# mask thrown
im_thrown_masked = np.empty_like(im_bcg_masked, dtype=np.uint8)
im_thrown_masked[:, :, 0] = cv2.bitwise_and(im_thrown[:, :, 0], mask_bcg)
im_thrown_masked[:, :, 1] = cv2.bitwise_and(im_thrown[:, :, 1], mask_bcg)
im_thrown_masked[:, :, 2] = cv2.bitwise_and(im_thrown[:, :, 2], mask_bcg)

# Calculate Difference
# Convert to gray scale
gray_bcg_masked = cv2.cvtColor(im_bcg_masked, cv2.COLOR_BGR2GRAY)
gray_thrown_masked = cv2.cvtColor(im_thrown_masked, cv2.COLOR_BGR2GRAY)

# Calculate difference inside mask
diff_thred, gray1, gray2, diff = getDifference(gray_bcg_masked, gray_thrown_masked)

# Dialate and close
kernel = np.ones((5,5), np.uint8)
dilation_bcg = cv2.dilate(diff_thred, kernel, iterations=3)
closing_bcg = cv2.morphologyEx(dilation_bcg, cv2.MORPH_CLOSE, kernel)

# transform bcg2ref
height, width, channels = im_ref.shape
closing_ref = cv2.warpPerspective(closing_bcg, h_bcg2ref, (width, height))
im_thrown_ref = cv2.warpPerspective(im_thrown, h_bcg2ref, (width, height))

# crop out object
closing_ref_cropped, left, right, top, bottom, contours = cropOutObject(closing_ref)

# Calculate Fitted Line
pt1, pt2 = getLine(closing_ref_cropped)
img = cv2.line(im_thrown_ref[top:bottom, left:right], pt1, pt2,(0,255,0),2)

# Calculate Apex
if len(contours) > 1:
    print("error: len( contours_cropped )  1")
arg = np.argmin(contours[0][:, 0, 0], axis=0)
# print(contours[0][:, 0])

line = cfg.getLinearFunctionCartesian(pt1[0], pt1[1], pt2[0], pt2[1])
apex_cropped_x = contours[0][:, 0][arg, 0] - left
apex_cropped = (apex_cropped_x, int(line(apex_cropped_x)))
print(apex_cropped)
outfile = cv2.circle(im_thrown_ref[top:bottom, left:right], apex_cropped,  10, (0, 0, 255), 5)

# outfile = cv2.drawContours(im_thrown_ref, contours, -1, (0,255,0), 3)


# Write inv homo to disk.
outFilename = "outputs/test_output.jpg"
print("Saving detected difference image : ", outFilename)
cv2.imwrite(outFilename, outfile)






# cv2.imshow('wtf', im_ref)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print('exitting image')