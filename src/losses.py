import torch
import torch.nn as nn

def rdrop_loss(logits1, logits2, targets, alpha = 1.0):
    mse_loss_combined = nn.MSELoss()(logits1, targets) + nn.MSELoss()(logits2, targets)
    rdrop_loss = alpha * nn.MSELoss()(logits1, logits2)

    return mse_loss_combined + rdrop_loss

def pearson(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_den = torch.clamp(r_den, min=1e-6)
    r_val = r_num / r_den
    r_val = torch.clamp(r_val, min=-1., max=1.)
    return r_val


def pearson_loss(pred, target):
    return 1. - pearson(pred, target)

def rank_loss(logits, targets):
    return pearson_loss(logits.view(-1), targets.view(-1))