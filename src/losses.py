import numba
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td
from catboost import MultiTargetCustomObjective

from math import log, exp, log1p, sqrt

@numba.jit
def sigmoid(x):
  return 1 / (1 + exp(-x))

@numba.jit
def softplus(x):
    return log1p(exp(-abs(x))) + max(x, 0)

def mape_loss(prediction, target, reduction="mean"):
    mask = target != 0
    diff = (prediction - target).abs()
    loss = torch.zeros_like(
        prediction, dtype=prediction.dtype,
        device=prediction.device
    )
    loss[mask] = diff[mask] / target[mask].abs()
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError(
            f"{reduction} mode is not supported yet. "
            f"Available reduction modes: \"mean\", \"sum\" and \"none\"."
        )
    return loss * 100


def mape_loss_wo_zeros(prediction, target, reduction="mean"):
    mask = target != 0
    diff = (prediction - target).abs()
    loss = torch.zeros_like(
        prediction, dtype=prediction.dtype,
        device=prediction.device
    )
    loss[mask] = diff[mask] / target[mask].abs()
    if reduction == "mean":
        loss = loss[mask].mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError(
            f"{reduction} mode is not supported yet. "
            f"Available reduction modes: \"mean\", \"sum\" and \"none\"."
        )
    return loss * 100

def ziln_loss(logits, target, reduce="mean"):
    is_positive = (target > 0).float()
    positive_logits = logits[:, :1]
    classification_loss = nn.BCEWithLogitsLoss()(
        positive_logits.flatten(), is_positive.flatten())
    loc = logits[:, 1:2]
    scale = torch.maximum(F.softplus(logits[:, 2:]), torch.ones_like(logits[:, 2:]) * 1e-5)
    safe_labels = is_positive * target + (1 - is_positive) * torch.ones_like(target)
    regression_loss = -torch.mean(
        is_positive * td.LogNormal(loc=loc, scale=scale).log_prob(safe_labels))
    return classification_loss + regression_loss


class MultiOutputLogNormal(MultiTargetCustomObjective):
    def calc_ders_multi(self, approx, target, weight):
        """
        :param approx: probability, mu and sigma
        :param target:
        :param weight:
        :return:
        """
        if target[0] == 0:
            der1 = [0.0, 0.0, 0.0]
        else:
            eps = 1e-6
            prob, mu, scale = approx
            prob = max(sigmoid(prob), eps)
            prob = min(prob, 1 - eps)
            mu_pred = approx[0]
            sigma_pred = max(softplus(approx[1]), sqrt(eps))
            y_true = target[0]

            dLdmu = (log(y_true) - mu_pred) / sigma_pred ** 2
            dLdsigma = - 1 / sigma_pred + ((log(y_true) - mu_pred) ** 2) / sigma_pred ** 3
            dLdpred = (y_true - prob) / (prob * (1 - prob))
            der1 = [dLdpred, dLdmu, dLdsigma]

        w = weight if weight is not None else 1.0
        der2 = [-w for i in range(len(approx))]

        return (der1, der2)