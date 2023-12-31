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
        weights = []
        for i in range(1, len(oofs) + 1):
            weights.append([trial.suggest_float(f'w{i}_{j}', -0.5, 2.) for j in range(2)])
        weights = np.array(weights)
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
    # # Human readable model oofs
    # oof_v1: deberta v3 large 1024 - cv 0.497
    # oof_v2: deberta v3 large 512
    # oof_v3: deberta v3 base 512
    # oof v4: roberta large 512
    # oof v5: electra large 512
    # oof v6: deberta v3 large 1800
    # oof v7: deberta v3 large 1024 mse - cv 0.495
    # oof v8: deberta large mnli 512 mcrmse
    # oof v9: deberta v3 large 1024 mean pool - cv 0.495
    # oof v10: deberta v3 large 1024 lstm pool
    # oof v11: deberta v3 large 1024 cls pool - cv 0.479 - max position embeddings 1024 (max len)
    # oof v12: deberta v3 large 1024 mean pool - cv 0.496 - max position embeddings 1024 (max len)
    # oof v13: deberta v3 large 1024 concat pool - cv 0.485 - max position embeddings 1024 (max len)
    # OOF paths
    oof_paths = [
        # "../output/oof_v1.csv",
        # "../output/oof_v2.csv",
        # "../output/oof_v3.csv",
        # "../output/oof_v4.csv",
        # "../output/oof_v5.csv",
        # "../output/oof_v7.csv",
        # "../output/oof_v8.csv",
        # "../output/oof_v9.csv",
        "../output/oof_v11.csv",
        # "../output/oof_v12.csv",
        "../output/oof_v13.csv"
        # "../output/oof_v10.csv",
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

    if avg_score < best_scores[best_idx]:
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

        final_oof = oofs[0].copy()
        for col in target_columns:
            final_oof[col] = 0
        
        for i in range(len(oofs)):
            if i == 0:
                final_oof[target_columns] = oofs[i][list(target_columns)].values
            else:
                final_oof[target_columns] += oofs[i][list(target_columns)].values

        final_oof[target_columns] /= len(oofs)

        final_oof.to_csv("../output/oof.csv", index=False)
    else:

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
        weights = []
        for i in range(1, len(oofs) + 1):
            weights.append([best_param[f'w{i}_{j}'] for j in range(2)])
        weights = np.array(weights)
        for i in range(len(oofs)):
            if i == 0:
                outputs = weights[i, :] * oofs[i][list(target_columns)].values
            else:
                outputs += weights[i, :] * oofs[i][list(target_columns)].values

        oof_df = oofs[0].copy()
        oof_df[list(target_columns)] = outputs
        # Saving the new oof file for training lgb model
        oof_df.to_csv("../output/oof.csv", index=False)
