import cv2
import numpy as np
from numpy.linalg import inv
import constraints
import detect_diff
import draw_dartboard
import board_config_cb as cfg
import align_Images

import importlib
importlib.reload(constraints)
importlib.reload(detect_diff)
importlib.reload(draw_dartboard)
importlib.reload(cfg)
importlib.reload(align_Images)

from constraints import SCORES
from detect_diff import getDifference, getLinePts, cropOutObject
from draw_dartboard import draw_regions
from align_Images import alignImages


# Read reference image
ref_maskedFilename = "resources/masked_shomen_cb.jpg"
print("Reading reference image : ", ref_maskedFilename)
im_ref_masked = cv2.imread(ref_maskedFilename, cv2.IMREAD_COLOR)

# Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
im_ref = cv2.imread(refFilename, cv2.IMREAD_COLOR)
tx = 0
ty = 0
im_ref_drawn = draw_regions(im_ref, tx, ty, cfg)

# Read mask
maskFilename = "resources/mask_shomen_cb_regions.jpg"
print("Reading mask image : ", maskFilename)
mask_ref = cv2.imread(maskFilename, cv2.IMREAD_COLOR)
mask_ref = mask_ref[:, :, 0]
# Read mask
mask_alignFilename = "resources/mask_shomen_cb.jpg"
print("Reading mask image : ", mask_alignFilename)
mask_ref_align = cv2.imread(mask_alignFilename, cv2.IMREAD_COLOR)
mask_ref_align = mask_ref_align[:, :, 0]



# mask reference
im_ref_masked  = np.empty_like(im_ref, dtype=np.uint8) #im_bcg.copy()
im_ref_masked [:, :, 0] = cv2.bitwise_and(im_ref[:, :, 0], mask_ref_align)
im_ref_masked [:, :, 1] = cv2.bitwise_and(im_ref[:, :, 1], mask_ref_align)
im_ref_masked [:, :, 2] = cv2.bitwise_and(im_ref[:, :, 2], mask_ref_align)
print(mask_ref.shape)

# VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(0)
ret, frame = capture.read()
frame = frame[:, :720]

count = 0
while True:
    ret, frame = capture.read()
    frame = frame[:, :720]
    cv2.imshow('Raw Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        # Read background image
        print("Reading reference image : ")
        im_bcg = frame


        print("Aligning images ...")
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        im_bcg2ref, h_bcg2ref = alignImages(im_bcg, im_ref_masked)
        h_ref2bcg = np.linalg.inv(h_bcg2ref)
        print("... homography attained")

        height, width, channels = im_bcg.shape
        mask_bcg = cv2.warpPerspective(mask_ref, h_ref2bcg, (width, height))

        # mask background
        im_bcg_masked = np.empty_like(im_bcg, dtype=np.uint8) #im_bcg.copy()
        im_bcg_masked[:, :, 0] = cv2.bitwise_and(im_bcg[:, :, 0], mask_bcg)
        im_bcg_masked[:, :, 1] = cv2.bitwise_and(im_bcg[:, :, 1], mask_bcg)
        im_bcg_masked[:, :, 2] = cv2.bitwise_and(im_bcg[:, :, 2], mask_bcg)

        cv2.imshow("Masked Background", im_bcg_masked)


    if cv2.waitKey(1) & 0xFF == ord('t'):
        print("key: \'t\' pressed!")
        # Read dart
        im_thrown = frame
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

        # transform thrown bcg2ref
        height, width, channels = im_ref_masked.shape
        closing_ref = cv2.warpPerspective(closing_bcg, h_bcg2ref, (width, height))
        im_thrown_ref = cv2.warpPerspective(im_thrown_masked, h_bcg2ref, (width, height))
        tx = 0
        ty = 0
        im_thrown_ref_drawn = draw_regions(im_thrown_ref, tx, ty, cfg)

      # crop out object
        closing_ref_cropped, left, right, top, bottom, contours, max_index = cropOutObject(closing_ref)

        # Calculate Fitted Line
        pt1, pt2 = getLinePts(closing_ref_cropped)
        im_thrown_ref_cropped = im_thrown_ref[top:bottom, left:right]
        cv2.imshow("detected object", im_thrown_ref_cropped)
        im_thrown_ref_cropped = cv2.line(im_thrown_ref_cropped, pt1, pt2,(0,255,255),2)
        cv2.imshow("detected object", im_thrown_ref_cropped)
        cts, hierarchy = cv2.findContours(closing_ref_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        im_thrown_ref_cropped = cv2.drawContours(im_thrown_ref_cropped, cts, -1, (0,255,0), 2 )
        cv2.imshow("detected object", im_thrown_ref_cropped)
        # Calculate Apex
        # get argument of contour with minimum x coordinate
        arg = np.argmin(contours[max_index][:, 0, 0], axis=0)
        # print(contours[0][:, 0])

        line = cfg.getLinearFunctionCartesian(pt1[0], pt1[1], pt2[0], pt2[1])
        apex_cropped_x = contours[max_index][:, 0][arg, 0] - left
        apex_cropped = (apex_cropped_x, int(line(apex_cropped_x)))
        cv2.circle(im_thrown_ref_drawn[top:bottom, left:right], apex_cropped,  10, (0, 0, 255), 5)

        # Calculate Apex
        arg = np.argmin(contours[0][:, 0, 0], axis=0)

        line = cfg.getLinearFunctionCartesian(pt1[0], pt1[1], pt2[0], pt2[1])
        apex_cropped_x = contours[0][:, 0][arg, 0] - left
        apex_cropped = (apex_cropped_x, int(line(apex_cropped_x)))
        cv2.imshow("detected object", im_thrown_ref_cropped)
        outfile = cv2.circle(im_thrown_ref_cropped, apex_cropped,  10, (0, 0, 255), 5)

        im_thrown_ref_cmasked = np.empty_like(im_ref, dtype=np.uint8) #im_bcg.copy()
        im_thrown_ref_cmasked [:, :, 0] = cv2.bitwise_and(im_ref[:, :, 0], closing_ref)
        im_thrown_ref_cmasked [:, :, 1] = cv2.bitwise_and(im_ref[:, :, 1], closing_ref)
        im_thrown_ref_cmasked [:, :, 2] = cv2.bitwise_and(im_ref[:, :, 2], closing_ref)
        im_thrown_ref_cmasked_cropped = im_thrown_ref_cmasked[top:bottom, left:right]
        cv2.imshow("detected object masked", im_thrown_ref_cmasked_cropped)
        im_thrown_ref_cmasked_cropped = cv2.line(im_thrown_ref_cmasked_cropped, pt1, pt2,(0,255,255),1)

        im_thrown_ref_cropped
        cv2.imshow("detected object", im_thrown_ref_cropped)
        cv2.imshow("detected object masked", im_thrown_ref_cmasked_cropped)
        cv2.imshow("Detected Apex", outfile)

        apex_ref = (apex_cropped[0] + left, apex_cropped[1] + top)
        outfile = cv2.circle(im_thrown_ref_drawn , apex_ref,  10, (0, 0, 255), 5)
        cv2.imshow("on score board", outfile)
        score = cfg.getScore(apex_ref)
        print(score, "!!!!!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\'q\' detected, exiting loop.")
        break
    count += 1
    print(".", end="")
    if count % 100 == 0:
        print(".")


capture.release()
cv2.destroyAllWindows()