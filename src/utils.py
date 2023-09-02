import numpy as np
import torch
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_mcrmse(preds, labels):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    if len(col_rmse) > 1:
        mcrmse = np.mean(col_rmse)

        return {
            "content_rmse": col_rmse[0],
            "wording_rmse": col_rmse[1],
            "mcrmse": mcrmse,
        }
    else:
        mcrmse = col_rmse[0]

        return {
            "mcrmse": mcrmse,
        }
    
class MeanPooling(nn.Module):
    """Mean pooling representation"""
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class LSTMPooling(nn.Module):
    """LSTM Pooling representation"""
    def __init__(self, hidden_size = 768):
        super(LSTMPooling, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size//2, batch_first=True, bidirectional=True)
        self.mean_pooling = MeanPooling()

    def forward(self, last_hidden_state, attention_mask):
        last_hidden_state, (_, _) = self.lstm(last_hidden_state)
        mean_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        return mean_embeddings