import numpy as np
import constraints
import importlib
import dp
import collections
import board_config_cb as cfgc ##n
import draw_dartboard
import pickle


importlib.reload(constraints)
importlib.reload(dp)
importlib.reload(cfgc) ##
importlib.reload(draw_dartboard)

from constraints import SCORES
from numpy.random import *
from matplotlib import pyplot as plt
import seaborn as sns


def cm2pix(cm):
    result = cm / 39.4 * cfgc.DOUBLE_OUT_R * 2
    return result


#ガウス二次元確率密度を返す関数
def getNormal(in_mu, in_sigma):
    #行列式
    det = np.linalg.det(in_sigma)
    #逆行列
    inv_sigma = np.linalg.inv(in_sigma)
    def f(xy):
        xy = np.array(xy)
        x_c = xy - in_mu
        return np.exp(- x_c.dot(inv_sigma).dot(x_c[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(det))
    return f


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable):
            yield from flatten(el)
        else:
            yield el


# # upload pickle

# board_description_str, board_description_int = draw_dartboard.getScoreDescription(img, cfgc)
# filename = 'dp_sets'
# outfile = open(filename, 'wb')
# pickle.dump(dp_sets, outfile)
# outfile.close()
# filename = 'board_description_str'
# outfile = open(filename, 'wb')
# pickle.dump(board_description_str, outfile)
# outfile.close()
# # load pickle
filename = 'dp_sets'
infile = open(filename, 'rb')
dp_sets = pickle.load(infile)
infile.close()
filename = 'board_description_int'
infile = open(filename, 'rb')
board_description_int = pickle.load(infile)
infile.close()
filename = 'board_description_str'
infile = open(filename, 'rb')
board_description_str = pickle.load(infile)
infile.close()




#######################
import cv2
# Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#######################

center_of_regions = cfgc.getCenterOfRegions()
center_of_regions_scores = cfgc.getCenterOfRegionsScores(center_of_regions)



def p(score_int_hit, region_aim, mu, sigma):
    region_args = np.argwhere(board_description_int == score_int_hit)
    # print("args for ", score_int_hit, " is: ", region_args.shape, end="")
    xy_aim = center_of_regions[region_aim]
    normal = getNormal(xy_aim, sigma)
    # print(xy_aim)
    # print(normal(xy_aim))
    result = 0
    for i in range(len(region_args)):
        # print(normal(region_args[i]))
        result += normal(region_args[i])
    # print("   ... result: p(",score_int_hit, region_aim, ") = ", result[0])
    return result

def getOptimFromScoreString(remaing, mu,  sigma):
    strategies = dp_sets[remaing]

    # get (x, y) of possible targets
    possible_targets_scores = list(set(list(flatten(strategies))))
    possible_targets = []
    for key, value in center_of_regions_scores.items():
        for i in range(len(possible_targets_scores)):
            if value == possible_targets_scores[i]:
                possible_targets.append(key)
    # print(possible_targets)

    # calculate argmin
    strategies_list = list(map(list, strategies))
    strategies_lengths = list(map(len, strategies))
    best_target = ""
    loss_min = np.inf
    worst_target = ""
    loss_max = 0
    losses_for_target = {}
    for d in possible_targets:
        loss = 0
        for k in range(len(strategies)):
            loss_sub = 0
            for m in range(strategies_lengths[k]):
                loss_sub += p(strategies_list[k][m], d, mu, sigma)
            # print("loss_sub = ", loss_sub)
            loss_sub *= strategies_lengths[k]
            loss += loss_sub
        # print("loss = ", loss)
        losses_for_target[d] = loss
        if loss < loss_min:
            # print("loss = ", loss, " loss_min = ", loss_min, "  >>>> target updated to: ", d)
            loss_min = loss
            best_target = d
        if loss > loss_max:
            # print("loss = ", loss, " loss_min = ", loss_min, "  >>>> target updated to: ", d)
            loss_max = loss
            worst_target = d
    return best_target, worst_target, losses_for_target


mu = [-5, -5]
sigma = [[10, 0], [0, 8]]

mu_pix = [cm2pix(-5), cm2pix(-5)]
sigma_pix = [[cm2pix(100), 30], [30, cm2pix(80)]]

# 2次元正規乱数生成
values = multivariate_normal(mu_pix, sigma_pix, 3)

# 散布図
sns.jointplot(values[:,0], values[:,1])
plt.show()

mu = np.mean(values, axis=0)
sigma = np.cov(np.transpose(values))
print(mu)
print(sigma)

import time
start = time.time()

best_target, worst_target, losses_for_target = getOptimFromScoreString(6, mu, sigma)
print(len(losses_for_target))
t = time.time() - start

print("t = ", t)
range_of_losses = losses_for_target[best_target] - losses_for_target[worst_target]

# # Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)


img = draw_dartboard.draw_regions(img, 0, 0, cfgc)
img_mu = img.copy()

center_of_regions = cfgc.getCenterOfRegions()

def getTargetForMu(mu, target):
    return tuple(np.array(value, dtype=int) + np.array(mu, dtype=int))

for key, value in center_of_regions.items():
    # print(key, value)
    color = (0, 0, 0)
    if key in losses_for_target:
        loss = losses_for_target[key]
        worst_loss = losses_for_target[worst_target]
        rate = 255 - int((loss - worst_loss) / range_of_losses * 255)
        color = (0, rate, 255)
    else: color = (0, 0, 0)
    img = cv2.circle(img, value, 5, color, 3)
    img_mu = cv2.circle(img_mu, getTargetForMu(mu, value), 5, color, 3)
    if key == best_target:
        img = cv2.circle(img, value, 10, (0,255,0), 5)
        img_mu = cv2.circle(img_mu, getTargetForMu(mu, value), 10, (0,255,0), 5)

# # Write drawn image to disk.
OUT_FILENAME = "outputs/draw_board_and_centers_cb.jpg"
print("Saving aligned image : ", OUT_FILENAME)
cv2.imwrite(OUT_FILENAME, img)
# # Write drawn image to disk.
OUT_FILENAME = "outputs/draw_board_and_centers_mu_cb.jpg"
print("Saving aligned image : ", OUT_FILENAME)
cv2.imwrite(OUT_FILENAME, img_mu)

