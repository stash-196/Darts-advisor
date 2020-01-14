
import numpy as np
import constraints
import importlib
importlib.reload(constraints)
from constraints import SCORES

COLOR = (255, 255, 255)
THICKNESS = 1

CENTER = (558, 306)
BULL_IN_R = 15
BULL_OUT_R = 31
TRIPPLE_IN_R = 133
TRIPPLE_OUT_R = 153
DOUBLE_IN_R = 216
DOUBLE_OUT_R = 240
BOAED_EDGE_R = 290
MASK_EDGE_R = 360

MASK_EDGE_BOTTOM = 597
MASK_EDGE_OFFSET = 64

FIRST_LINE_END_X = 515
FIRST_LINE_END_Y = 70
THETA_ZERO = np.arctan((FIRST_LINE_END_Y - CENTER[1])/(FIRST_LINE_END_X - CENTER[0]))
# print(THETA_ZERO)

PI_SEGMENTS = np.linspace(-np.pi, np.pi, 21, endpoint=True)
# print(PI_SEGMENTS)

def getLinearFunctionCartesian(x1, y1, x2, y2):
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a * x1
    return lambda x: a * x + b

SCORES_LIST = []
for key, value in SCORES.items():
    temp = [key,value]
    SCORES_LIST.append(temp)
# print(SCORES_LIST)

def getScore(target):
    target = np.array(target)
    center = np.array(CENTER)
    relative = target - center
    r = np.linalg.norm(relative)
    theta = np.arctan2(relative[1], relative[0]) - THETA_ZERO
    if theta < -np.pi: theta += 2*np.pi
    # Bullseyes
    if r < BULL_IN_R:
        return "bull_in"
    elif r < BULL_OUT_R:
        return "bull_out"
    else:
        # which segment
        for i in range(PI_SEGMENTS.size-1):
            if PI_SEGMENTS[i] <= theta < PI_SEGMENTS[i+1]:
                # Is it in DOUBLE or TRIPPLE?
                if DOUBLE_IN_R <= r <= DOUBLE_OUT_R:
                    return SCORES_LIST[i + 20][0]
                elif TRIPPLE_IN_R <= r <= TRIPPLE_OUT_R:
                    return SCORES_LIST[i + 40][0]
                elif r <= DOUBLE_IN_R:
                    return SCORES_LIST[i][0]
                else:
                    return "?"
        print("i=", i)
        print("i, j = ", target)
        print("theta = ", theta/np.pi*180)
        print("ERROR: \'board_config.py\'score description was unattainnable")
        return "!"

def getRelativeRadian(theta):
    result = theta + THETA_ZERO
    if result < -np.pi: result += 2*np.pi
    return result

def getCenterOfGravity(center, phi0, phi1, r0, r1):
    # Calculate Center of Gravity in inner
    relative0  = np.array([r0 * np.cos(phi0), r0 * np.sin(phi0)], dtype=np.int)
    vertex0 = center + relative0

    relative1  = np.array([r0 * np.cos(phi1), r0 * np.sin(phi1)], dtype=np.int)
    vertex1 = center + relative1

    relative2  = np.array([r1 * np.cos(phi0), r1 * np.sin(phi0)], dtype=np.int)
    vertex2 = center + relative2

    relative3  = np.array([r1 * np.cos(phi1), r1 * np.sin(phi1)], dtype=np.int)
    vertex3 = center + relative3

    result = np.zeros(2, dtype=np.int)
    result = (int((vertex0[0] + vertex1[0] + vertex2[0] + vertex3[0]) / 4), int((vertex0[1] + vertex1[1] + vertex2[1] + vertex3[1]) / 4))
    return result


def getCenterOfRegions():
    center = np.array(CENTER)
    # relative = target - center
    # r = np.linalg.norm(relative)
    # theta = np.arctan2(relative[1], relative[0]) - THETA_ZERO
    # if theta < -np.pi: theta += 2*np.pi
    half_segment = np.pi / 20

    center_of_regions = {}
    scores = constraints.SCORES
    i = 0
    for key, value in scores.items():
        # print("i: ", i)
        if 'bull' in key:
            cg = CENTER
            center_of_regions[key] = cg
        elif 'D' in key:
            phi0 = getRelativeRadian(PI_SEGMENTS[i - 20])
            phi1 = getRelativeRadian(PI_SEGMENTS[i - 20 + 1])
            r0 = DOUBLE_IN_R
            r1 = DOUBLE_OUT_R
            cg = getCenterOfGravity(center, phi0, phi1, r0, r1)

            center_of_regions[key] = cg
        elif 'T' in key:
            phi0 = getRelativeRadian(PI_SEGMENTS[i - 40])
            phi1 = getRelativeRadian(PI_SEGMENTS[i - 40 + 1])
            r0 = TRIPPLE_IN_R
            r1 = TRIPPLE_OUT_R
            cg = getCenterOfGravity(center, phi0, phi1, r0, r1)

            center_of_regions[key] = cg
        else:
            phi0 = getRelativeRadian(PI_SEGMENTS[i])
            phi1 = getRelativeRadian(PI_SEGMENTS[i + 1])
            r0 = BULL_OUT_R
            r1 = TRIPPLE_IN_R
            cg = getCenterOfGravity(center, phi0, phi1, r0, r1)

            key_for_inner = key + '_inner'
            center_of_regions[key_for_inner] = cg

            # Calculate Center of Gravity in outer
            r0 = TRIPPLE_OUT_R
            r1 = DOUBLE_IN_R
            cg = getCenterOfGravity(center, phi0, phi1, r0, r1)
            key_for_outer = key + '_outer'
            center_of_regions[key_for_outer] = cg
        i += 1
    return center_of_regions

def getCenterOfRegionsScores(center_of_regions):
    center_of_regions_scores = center_of_regions.copy()
    for key, value in center_of_regions_scores.items():
        if "_inner" in key: 
            score_string = key.replace("_inner","")
            center_of_regions_scores[key] = int(score_string)
        if "_outer" in key: 
            score_string = key.replace("_outer","")
            center_of_regions_scores[key] = int(score_string)
        if "_D" in key: 
            score_string = key.replace("_D","")
            center_of_regions_scores[key] = int(score_string) * 2
        if "_T" in key: 
            score_string = key.replace("_T","")
            center_of_regions_scores[key] = int(score_string) * 3
    center_of_regions_scores["bull_in"] = 50
    center_of_regions_scores["bull_out"] = 25
    return center_of_regions_scores


# def getAllLinearFunctions():
#     line_functions = {}
#     for key, value in LINE_ENDPOINTS.items():
#         PTS = LINE_ENDPOINTS[key]
#         line_functions[key] = getLinearFunction(PTS[0][0], PTS[0][1], PTS[1][0], PTS[1][1])
#     return line_functions

# PTS = LINE_ENDPOINTS['5-20_17-3']
# print(PTS)
# f = getLinearFunction(PTS[0][0], PTS[0][1], PTS[1][0], pts[1][1])
# print(f(264))

# lfs = getAllLinearFunctions()
# print(lfs['6-10_11-14'](65))