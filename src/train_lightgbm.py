import pandas as pd
from preprocesser import Preprocessor
from tqdm import tqdm
import lightgbm as lgb
from utils import compute_mcrmse
import logging
import optuna
from functools import partial
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

# Target columns and columns to drop while training
targets = ["content", "wording"]

drop_columns = ["fold", "student_id", "prompt_id", "text", "corrected_text",
                "prompt_question", "prompt_title", 
                "prompt_text"
               ] + targets

def train_lgb(prompts_path, summaries_path, model_name, oof_file_path):
    prompts = pd.read_csv(prompts_path)
    summaries = pd.read_csv(summaries_path)
    if isinstance(oof_file_path, list):
        oof_df = [pd.read_csv(oof_path).rename(columns = {"content" : f"pred_content_{idx}", "wording" : f"pred_wording_{idx}"}).drop(columns = ["prompt_id"]) if idx == 1 else  pd.read_csv(oof_path).rename(columns = {"content" : f"pred_content_{idx}", "wording" : f"pred_wording_{idx}"}).drop(columns = ["prompt_id", "fold", "student_id"]) for idx, oof_path in enumerate(oof_file_path, 1)]
        oof_df = pd.concat(oof_df, axis = 1)
    else:
        oof_df = pd.read_csv(oof_file_path).rename(columns = {"content" : "pred_content", "wording" : "pred_wording"}).drop(columns = ["prompt_id"])
    preprocessor = Preprocessor(model_name = model_name)
    df = preprocessor.run(prompts, summaries, mode="train")
    df = df.merge(oof_df, on = "student_id")

    models = []
    for target in targets:
        for fold in range(4):
            print(f"Training {target} for fold {fold}")
            X_train_cv = df[df["fold"] != fold].drop(columns=drop_columns)
            train_columns = X_train_cv.columns
            y_train_cv = df[df["fold"] != fold][target]

            X_eval_cv = df[df["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = df[df["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            # params = {
            #         'boosting_type': 'gbdt',
            #         'random_state': 42,
            #         'objective': 'regression',
            #         'metric': 'rmse',
            #         'learning_rate': 0.05,
            # }
            params = {
                'boosting_type': 'gbdt',
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': 5, 'learning_rate': 0.08185194910306416, 'lambda_l1': 7.249736223975043e-08, 'lambda_l2': 2.5435724074500937e-07, 'num_leaves': 20,
                'verbose': -1,
            }
            
            evaluation_results = {}
            model = lgb.train(params,
                            num_boost_round=10000,
                                #categorical_feature = categorical_features,
                            valid_names=['train', 'valid'],
                            train_set=dtrain,
                            valid_sets=dval,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=30, verbose=True),
                                lgb.log_evaluation(100),
                                lgb.callback.record_evaluation(evaluation_results)
                                ],
                            )
            
            model.save_model(f'../output/lgbm_{target}_fold_{fold}.txt', num_iteration=model.best_iteration)

    ## Calculating final OOF Score
    df_copy = df.copy()
    for target in targets:
        df_copy[f"final_pred_{target}"] = 0
        for fold in range(4):
            model = lgb.Booster(model_file = f'../output/lgbm_{target}_fold_{fold}.txt')
            test_columns = df[df["fold"] == fold].drop(columns=drop_columns).columns
            assert (train_columns == test_columns).all()
            df_copy.loc[df_copy["fold"] == fold, f"final_pred_{target}"] = model.predict(df[df["fold"] == fold].drop(columns=drop_columns))
    
    score = compute_mcrmse(df_copy[[f"final_pred_{target}" for target in targets]].values, df_copy[targets].values)
    return score

def train_lgb_hparam(params, df):

    models = []
    for target in targets:
        for fold in range(4):
            print(f"Training {target} for fold {fold}")
            X_train_cv = df[df["fold"] != fold].drop(columns=drop_columns)
            train_columns = X_train_cv.columns
            y_train_cv = df[df["fold"] != fold][target]

            X_eval_cv = df[df["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = df[df["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            evaluation_results = {}
            model = lgb.train(params,
                            num_boost_round=10000,
                                #categorical_feature = categorical_features,
                            valid_names=['train', 'valid'],
                            train_set=dtrain,
                            valid_sets=dval,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=30, verbose=True),
                                lgb.log_evaluation(100),
                                lgb.callback.record_evaluation(evaluation_results)
                                ],
                            )
            
            model.save_model(f'../output/lgbm_{target}_fold_{fold}.txt', num_iteration=model.best_iteration)

    ## Calculating final OOF Score
    df_copy = df.copy()
    for target in targets:
        df_copy[f"final_pred_{target}"] = 0
        for fold in range(4):
            model = lgb.Booster(model_file = f'../output/lgbm_{target}_fold_{fold}.txt')
            test_columns = df[df["fold"] == fold].drop(columns=drop_columns).columns
            assert (train_columns == test_columns).all()
            df_copy.loc[df_copy["fold"] == fold, f"final_pred_{target}"] = model.predict(df[df["fold"] == fold].drop(columns=drop_columns))
    
    score = compute_mcrmse(df_copy[[f"final_pred_{target}" for target in targets]].values, df_copy[targets].values)
    return score

def objective(trial, df):
    # params = {
    #     'metric': 'rmse', 
    #     'random_state': 41,
    #     'n_estimators': 20000,
    #     'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
    #     'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    #     'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
    #     'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
    #     'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02, 0.05]),
    #     'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
    #     'num_leaves' : trial.suggest_int('num_leaves', 2, 1000),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    #     'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
    #     'boosting_type': 'gbdt',
    #     'verbose': -1,
    # } 
    max_depth = trial.suggest_int('max_depth', 2, 10)
    params = {
        'boosting_type': 'gbdt',
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': max_depth,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 2**max_depth - 1),
        'verbosity': -1  # Add this line to suppress warnings and info messages

    }
    score = train_lgb_hparam(params, df)
    mcrmse = score["mcrmse"]
    return mcrmse

def get_preprocessed_df(prompts_path, summaries_path, model_name, oof_file_path):
    prompts = pd.read_csv(prompts_path)
    summaries = pd.read_csv(summaries_path)
    if isinstance(oof_file_path, list):
        oof_df = [pd.read_csv(oof_path).rename(columns = {"content" : f"pred_content_{idx}", "wording" : f"pred_wording_{idx}"}).drop(columns = ["prompt_id"]) if idx == 1 else  pd.read_csv(oof_path).rename(columns = {"content" : f"pred_content_{idx}", "wording" : f"pred_wording_{idx}"}).drop(columns = ["prompt_id", "fold", "student_id"]) for idx, oof_path in enumerate(oof_file_path, 1)]
        oof_df = pd.concat(oof_df, axis = 1)
    else:
        oof_df = pd.read_csv(oof_file_path).rename(columns = {"content" : "pred_content", "wording" : "pred_wording"}).drop(columns = ["prompt_id"])
    preprocessor = Preprocessor(model_name = model_name)
    df = preprocessor.run(prompts, summaries, mode="train")
    df = df.merge(oof_df, on = "student_id")
    return df
    

if __name__ == "__main__":
    # # Human readable model oofs
    # oof_v1: deberta v3 large 1024
    # oof_v2: deberta v3 large 512
    # oof_v3: deberta v3 base 512
    # oof v4: roberta large 512
    # oof v5: electra large 512
    # oof v6: deberta v3 large 1800
    # oof v7: deberta v3 large 1024 mse 
    oof_file_path = [
        "../output/oof_v1.csv",
        "../output/oof_v2.csv",
        "../output/oof_v3.csv",
        "../output/oof_v4.csv",
        "../output/oof_v5.csv",
        "../output/oof_v7.csv"
    ]
    # oof_file_path = "../output/oof.csv"
    # df = get_preprocessed_df(prompts_path = "../data/prompts_train.csv", 
    #                         summaries_path = "../data/summaries_train.csv",
    #                         model_name = "microsoft/deberta-v3-large", 
    #                         oof_file_path= oof_file_path)
    # oof_score_deberta = compute_mcrmse(df[["pred_content_6", "pred_wording_6"]].values, df[["content", "wording"]].values)["mcrmse"]
    # print("OOF score model deberta: ", oof_score_deberta)
    # study = optuna.create_study(direction='minimize')
    # objective = partial(objective, df = df)
    # study.optimize(objective, n_trials=200)
    # print('Number of finished trials:', len(study.trials))
    # print('Best trial:', study.best_trial.params)
    
    score = train_lgb(
        prompts_path = "../data/prompts_train.csv", 
        summaries_path = "../data/summaries_train.csv",
        model_name = "microsoft/deberta-v3-large", 
        oof_file_path= oof_file_path
    )["mcrmse"]

    print("The CV score is: ", score)
