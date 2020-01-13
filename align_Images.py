# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import numpy as np

import plotly.graph_objects as go


import sys
# print(sys.path)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.05 #0.15
# %%
def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    slopes_points = []
    slopes = []
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        slope = (points2[i, 1] - points1[i, 1])/(points2[i, 0] + 720 - points1[i, 0])
        append = [slope, points1[i, :].tolist(), points2[i, :].tolist()]
        slopes_points.append(append)
        slopes.append(slope)
    slopes_points_sorted = sorted(slopes_points, key=lambda pts: pts[0])

    slopes_sorted = sorted(slopes)

    quartile_1, quartile_3 = np.percentile(slopes_sorted, [25, 75])
    iqr = quartile_3 - quartile_1
    # 下限
    lower_bound = quartile_1 - (iqr * 1.5)
    # 上限
    upper_bound = quartile_3 + (iqr * 1.5)
    # print(upper_bound, lower_bound)
    if len(slopes) != len(points1):
        print("Oh no ", len(slopes), len(points1))

    invalid = []
    for i in range(len(slopes)):
        slope = (points2[i, 1] - points1[i, 1])/(points2[i, 0] + 720 - points1[i, 0])
        if slope <=lower_bound or upper_bound <= slope:
            invalid.append(i)
    points1_wo_invalid = np.delete(points1, invalid, 0)
    points2_wo_invalid = np.delete(points2, invalid, 0)

    # x = np.random.randn(500)
    # fig = go.Figure(data=[go.Histogram(x=slopes_sorted_cropped, histnorm='probability')])



    # fig.show()
    # fig.write_image("outputs/slopes_trimmed.png")

    # Find homography
    h, mask = cv2.findHomography(points1_wo_invalid, points2_wo_invalid, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def identify_outliers(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    # 下限
    lower_bound = quartile_1 - (iqr * 1.5)
    # 上限
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.array(ys)[((ys > upper_bound) | (ys < lower_bound))]







# q_1 = 0
# for i in range(len(slopes_points_sorted)):
#     if slopes_points_sorted[i][0] > lower_bound:
#         q_1 = i
#         # print("q_1 = ", q_1)
#         break
# q_3 = q_1
# for i in range(q_1, len(slopes_points_sorted)):
#     if slopes_points_sorted[i][0] > upper_bound:
#         q_3 = i - 1
#         # print("q_3 = ", q_3)
#         break


# slopes_points_sorted_cropped = slopes_points_sorted[q_1:q_3]
# # slopes_sorted_cropped = np.array(slopes_points_sorted_cropped)[:, 0]

# slopes_points1_cropped = np.array(slopes_points_sorted_cropped)[:, 1]
# slopes_points2_cropped = np.array(slopes_points_sorted_cropped)[:, 2]

# points1_cropped = np.empty_like(points1)[q_1:q_3]
# points2_cropped = np.empty_like(points1)[q_1:q_3]
# for i in range(len(points1_cropped)):
#     points1_cropped[i, 0] = slopes_points1_cropped[i][0]
#     points1_cropped[i, 1] = slopes_points1_cropped[i][1]
#     points2_cropped[i, 0] = slopes_points2_cropped[i][0]
#     points2_cropped[i, 1] = slopes_points2_cropped[i][1]
# print(len(points1_cropped))