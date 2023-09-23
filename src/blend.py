import pandas as pd
import numpy as np
from utils import compute_mcrmse
import optuna
import time
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main(oof_paths):

    oofs = [
        pd.read_csv(oof_path) for oof_path in oof_paths
    ]

    target_columns = ['content', 'wording']

    # Loading the actual targets and prompt and summary file
    pdf = pd.read_csv("../data/prompts_train.csv")
    sdf = pd.read_csv("../data/summaries_train.csv")
    df = pdf.merge(sdf, on="prompt_id")

    for oof in oofs:
        assert all([x == y for x, y in zip(oof["student_id"].to_list(), df["student_id"].to_list())])
    targets = df[list(target_columns)].values

    def get_average_score():
        for i in range(len(oofs)):
            if i == 0:
                outputs = oofs[i][list(target_columns)].values
            else:
                outputs += oofs[i][list(target_columns)].values
        outputs /= len(oofs)
        score = compute_mcrmse(outputs, targets)["mcrmse"]
        return score

    avg_score = get_average_score()
    

    def objective(trial):
        w1 = [trial.suggest_float(f'w1_{i}', -0.5, 2.) for i in range(2)]
        w2 = [trial.suggest_float(f'w2_{i}', -0.5, 2.) for i in range(2)]
        w3 = [trial.suggest_float(f'w3_{i}', -0.5, 2.) for i in range(2)]
        w4 = [trial.suggest_float(f'w4_{i}', -0.5, 2.) for i in range(2)]
        w5 = [trial.suggest_float(f'w5_{i}', -0.5, 2.) for i in range(2)]
        weights = np.array([w1, w2, w3, w4, w5])
        for i in range(len(oofs)):
            if i == 0:
                outputs = weights[i, :] * oofs[i][list(target_columns)].values
            else:
                outputs += weights[i, :] * oofs[i][list(target_columns)].values
        score = compute_mcrmse(outputs, targets)["mcrmse"]
        return score

    best_params, best_scores = [], []
    for seed in range(5):

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=1000, n_jobs=4, show_progress_bar = True)

        best_params.append(study.best_params)
        best_scores.append(study.best_value)

    return best_params, best_scores, avg_score

if __name__ == "__main__":
    # Human readable model oofs
    # oof_v1: deberta v3 large 1024
    # oof_v2: deberta v3 large 512
    # oof_v3: deberta v3 base 512
    # oof v4: roberta large 512
    # oof v5: electra large 512
    # OOF paths
    oof_paths = [
        "../output/oof_v1.csv",
        "../output/oof_v2.csv",
        "../output/oof_v3.csv",
        "../output/oof_v4.csv",
        "../output/oof_v5.csv",
    ]
    best_params, best_scores, avg_score = main(oof_paths)

    best_idx = np.argmin(best_scores)
    print(f"Params: {best_params}")
    print(f"Scores: {best_scores}")
    print("***"* 50)
    print(f'\nThe best score is {best_scores[best_idx]}')
    print(f"The best params are {best_params[best_idx]}")
    print(f"The average score is {avg_score}")
    print("***"* 50)

    # Loading the actual targets and prompt and summary file
    pdf = pd.read_csv("../data/prompts_train.csv")
    sdf = pd.read_csv("../data/summaries_train.csv")
    df = pdf.merge(sdf, on="prompt_id")

    oofs = [
        pd.read_csv(oof_path) for oof_path in oof_paths
    ]

    for oof in oofs:
        assert all([x == y for x, y in zip(oof["student_id"].to_list(), df["student_id"].to_list())])

    target_columns = ['content', 'wording']

    best_param = best_params[best_idx]
    # best_param = {'w1_0': 0.6947302511269289, 'w1_1': 0.6639292761466956, 'w2_0': 0.5966504446856765, 'w2_1': 0.028682039728677812, 'w3_0': -0.2993895027547027, 'w3_1': -0.23320086908303095, 'w4_0': 0.2772219415312553, 'w4_1': 0.4656334430730368, 'w5_0': -0.3409878557307244, 'w5_1': 0.013978204062719768}
    w1 = [best_param[f'w1_{i}'] for i in range(2)]
    w2 = [best_param[f'w2_{i}'] for i in range(2)]
    w3 = [best_param[f'w3_{i}'] for i in range(2)]
    w4 = [best_param[f'w4_{i}'] for i in range(2)]
    w5 = [best_param[f'w5_{i}'] for i in range(2)]
    weights = np.array([w1, w2, w3, w4, w5])
    for i in range(len(oofs)):
        if i == 0:
            outputs = weights[i, :] * oofs[i][list(target_columns)].values
        else:
            outputs += weights[i, :] * oofs[i][list(target_columns)].values

    oof_df = oofs[0].copy()
    oof_df[list(target_columns)] = outputs
    # Saving the new oof file for training lgb model
    oof_df.to_csv("../output/oof.csv", index=False)
