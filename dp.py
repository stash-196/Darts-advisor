import numpy as np
import constraints

import importlib
importlib.reload(constraints)
from constraints import SCORES

def getDPSets(upper_bound):
    SCORE_VALUES = SCORES.values()
    SCORE_VALUES = list(set(SCORE_VALUES))
    np.full((2, 2), np.inf)
    MAX_TARGET = upper_bound
    dp_counts = np.full((len(SCORE_VALUES), MAX_TARGET + 1), np.inf)
    dp_counts[:, 0] = 0
    dp_min_counts = [0] * (MAX_TARGET + 1)
    dp_sets_ordered = [[[]]]
    dp_sets = [{()}]

    for j in range(1, MAX_TARGET + 1):
        sets_ordered = []
        for i in range(len(SCORE_VALUES)):
            remaining = j - SCORE_VALUES[i]
            if remaining >= 0:
                dp_counts[i, j] = dp_min_counts[remaining] + 1
                sets_ordered.append(sorted([SCORE_VALUES[i]] + dp_sets_ordered[remaining][0], reverse=True))
        dp_min_counts[j] = min(dp_counts[:, j])
        sets_ordered = sorted(sets_ordered, key=len, reverse=False)
        dp_sets_ordered.append(sets_ordered)
        sets = set(map(tuple, sets_ordered))
        sets_reordered = sorted(list(sets), key=len, reverse=False)
        dp_sets.append(sets_reordered)
        # print("j=", j, "-> appending: ", sets_reordered)
    return dp_sets, dp_min_counts
    # print("result: ", dp_sets)

# for t in range(len(dp_sets_ordered)):
#     print(t, ",", dp_min_counts[t], end=""), print(dp_sets_ordered[t])

getDPSets(23)