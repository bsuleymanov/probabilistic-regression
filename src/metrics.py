from scipy import stats

def spearmanr(prediction, target):
    return stats.spearmanr(target, prediction, nan_policy='raise')[0]

