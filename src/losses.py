import torch
import torch.nn as nn

def rdrop_loss(logits1, logits2, targets, alpha = 1.0):
    mse_loss_combined = nn.MSELoss()(logits1, targets) + nn.MSELoss()(logits2, targets)
    rdrop_loss = alpha * nn.MSELoss()(logits1, logits2)

    return mse_loss_combined + rdrop_loss