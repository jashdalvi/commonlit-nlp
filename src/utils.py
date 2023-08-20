import numpy as np

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