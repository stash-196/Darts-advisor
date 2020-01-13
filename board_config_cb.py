
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
        return 'bull_in'
    elif r < BULL_OUT_R:
        return 'bull_out'
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
                    return ''
        print("i=", i)
        print("i, j = ", target)
        print("theta = ", theta/np.pi*180)
        print("ERROR: \'board_config.py\'score description was unattainnable")
        return ''


def getCenterOfRegions():

    
    center_of_regions = {}
    scores = constraints.SCORES_ordered
    for key, value in scores.items():
        if 'D' in key:
            pass
        elif 'T' in key:
            pass
        else:
            vertex_1 = 

getCenterOfRegions()





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