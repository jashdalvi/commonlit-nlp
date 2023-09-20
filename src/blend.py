import pandas as pd
import numpy as np
from .utils import compute_mcrmse
import optuna
import time
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    # Add oof paths as necessary
    oof_paths = [
        "../output/oof_v1.csv",
        "../output/oof_v2.csv"
    ]

    oofs = [
        pd.read_csv(oof_path) for oof_path in oof_paths
    ]

    target_columns = ['content', 'wording']

    # Loading the actual targets and prompt and summary file
    pdf = pd.read_csv("../data/prompts_train.csv")
    sdf = pd.read_csv("../data/summaries_train.csv")
    df = pdf.merge(sdf, on="prompt_id")
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
        weights = np.array([w1, w2, w3])
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
    best_params, best_scores, avg_score = main()

    best_idx = np.argmax(best_scores)
    print(f"Params: {best_params}")
    print(f"Scores: {best_scores}")
    print("***"* 50)
    print(f'\nThe best score is {best_scores[best_idx]}')
    print(f"The best params are {best_params[best_idx]}")
    print(f"The average score is {avg_score}")
    print("***"* 50)


    # Saving final oof_file
    oof_paths = [
        "../output/oof_v1.csv",
        "../output/oof_v2.csv"
    ]

    oofs = [
        pd.read_csv(oof_path) for oof_path in oof_paths
    ]

    target_columns = ['content', 'wording']

    best_param = best_params[best_idx]
    w1 = [best_param[f'w1_{i}'] for i in range(2)]
    w2 = [best_param[f'w2_{i}'] for i in range(2)]
    w3 = [best_param[f'w3_{i}'] for i in range(2)]
    weights = np.array([w1, w2, w3])
    for i in range(len(oofs)):
        if i == 0:
            outputs = weights[i, :] * oofs[i][list(target_columns)].values
        else:
            outputs += weights[i, :] * oofs[i][list(target_columns)].values

    oof_df = oofs[0].copy()
    oof_df[list(target_columns)] = outputs
    # Saving the new oof file for training lgb model
    oof_df.to_csv("../output/oof.csv", index=False)
