
import numpy as np
import constraints
import importlib
importlib.reload(constraints)
from constraints import SCORES

COLOR = (255, 255, 255)
THICKNESS = 1

CENTER = (287, 270)
BULL_IN_R = 15
BULL_OUT_R = 30
TRIPPLE_IN_R = 127
TRIPPLE_OUT_R = 145
DOUBLE_IN_R = 207
DOUBLE_OUT_R = 228
BOAED_EDGE_R = 280
MASK_EDGE_R = 350

THETA_ZERO = np.arctan((45 - CENTER[1])/(264 - CENTER[0]))
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

def getScoreDescription(target):
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
                else:
                    return SCORES_LIST[i][0]
        print("i=", i)
        print("i, j = ", target)
        print("theta = ", theta/np.pi*180)
        print("ERROR: \'board_config.py\'score description was unattainnable")
        return ''


# --------------------------------------------
# test scoredescriptions
# sd = getScoreDescription([300, 100]) #20
# print(sd)
# sd = getScoreDescription([350, 100]) # 1
# print(sd)
# sd = getScoreDescription([400, 130]) #18
# print(sd)
# sd = getScoreDescription([450, 400]) #15
# print(sd)
# sd = getScoreDescription([400, 430]) #2
# print(sd)
# sd = getScoreDescription([335, 440]) #17
# print(sd)
# sd = getScoreDescription([270, 440]) #3
# print(sd)
# sd = getScoreDescription([220, 447]) #19
# print(sd)
# sd = getScoreDescription([693, 301])
# print(sd)
# sd = getScoreDescription([416, 97])
# print(sd)
# sd = getScoreDescription([370, 377])
# print(sd)
# sd = getScoreDescription([209, 382])
# print(sd)
# sd = getScoreDescription([76, 311])
# print(sd)
# sd = getScoreDescription([409, 325])
# print(sd)
# sd = getScoreDescription([481, 173])
# print(sd)



# LINE_ENDPOINTS = {
#     '5-20_17-3': [(264, 45), (312, 496)],
#     '20-1_3-19': [(338, 51), (238, 494)],
#     '1-18_19-7': [(402, 76), (172, 467)],
#     '18-4_7-16': [(456, 122), (114, 421)],
#     '4-13_16-8': [(498, 179), (77, 360)],
#     '13-6_8-11': [(514, 249), (60, 292)],
#     '6-10_11-14': [(514, 316), (65, 223)],
#     '10-15_14-9': [(486, 387), (90, 155)],
#     '15-2_9-12': [(442, 442), (136, 102)],
#     '2-17_12-5': [(385, 482), (198, 61)]
# }

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