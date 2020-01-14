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

mu = [cm2pix(-5), cm2pix(-5)]
sigma = [[cm2pix(100), 0], [0, cm2pix(80)]]

# 2次元正規乱数を1万個生成
values = multivariate_normal(mu, sigma, 3)

# # 散布図
# sns.jointplot(values[:,0], values[:,1])
# plt.show()

mu = np.mean(values, axis=0)
sigma = np.cov(np.transpose(values))
print(mu)
print(sigma)


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



board_description_str, board_description_int = draw_dartboard.getScoreDescription(img, cfgc)
filename = 'board_description_int'
outfile = open(filename, 'wb')
pickle.dump(board_description_int, outfile)
outfile.close()
filename = 'board_description_str'
outfile = open(filename, 'wb')
pickle.dump(board_description_str, outfile)
outfile.close()

#######################
import cv2
# Read reference image
refFilename = "resources/shomen_cb.jpeg"
print("Reading reference image : ", refFilename)
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#######################

center_of_regions = cfgc.getCenterOfRegions()
center_of_regions_scores = cfgc.getCenterOfRegionsScores(center_of_regions)

board_description_int.shape


def p(score_int_hit, region_aim):
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
    return result[0]

def getOptim(remaing):
    dp_sets, dp_counts = dp.getDPSets(60)
    strategies = dp_sets[remaing]
    possible_targets_scores = list(set(list(flatten(strategies))))
    print(possible_targets_scores)
    possible_targets = []
    for key, value in center_of_regions_scores.items():
        for i in range(len(possible_targets_scores)):
            if value == possible_targets_scores[i]:
                possible_targets.append(key)
    print(possible_targets)

    strategies_list = list(map(list, strategies))
    strategies_lengths = list(map(len, strategies))
    best_target = ""
    loss_min = np.inf
    for d in set(possible_targets):
        loss = 0
        for k in range(len(strategies)):
            loss_sub = 0
            for m in range(strategies_lengths[k]):
                loss_sub += p(strategies_list[k][m], d)
            # print("loss_sub = ", loss_sub)
            loss_sub *= strategies_lengths[k]
            loss += loss_sub
        # print("loss = ", loss)

        if loss < loss_min:
            # print("loss = ", loss, " loss_min = ", loss_min, "  >>>> target updated to: ", d)
            loss_min = loss
            best_target = d
    return best_target





optim_point = getOptim(6)
print(optim_point)
p(9, "bull_in")
