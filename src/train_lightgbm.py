import pandas as pd
from preprocesser import Preprocessor
from tqdm import tqdm
import lightgbm as lgb
from utils import compute_mcrmse
import logging
logging.basicConfig(level=logging.ERROR)
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

            params = {
                    'boosting_type': 'gbdt',
                    'random_state': 42,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'learning_rate': 0.05,
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