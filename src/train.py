import torch
import torch.nn as nn
import numpy as np
import os
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, pipeline
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error
from awp import AWP
import random
import time
from torch.utils import checkpoint
import math
import gc
from typing import Dict, List, Tuple
import codecs
import warnings
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from utils import AverageMeter, compute_mcrmse, MeanPooling, LSTMPooling, CLSPooling, MaxPooling, MeanMaxPooling, ConcatPooling, noisy_tune
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
import subprocess
from huggingface_hub import login, HfApi, hf_hub_download, snapshot_download, create_repo
import shutil
import json
from train_lightgbm import train_lgb
from utils import MeanPooling, LSTMPooling
from huggingface_hub import login
from losses import rdrop_loss, rank_loss
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
tqdm.pandas()
load_dotenv()

# declare the two GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# avoids some issues when using more than one worker
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Initialize hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()

# Dict for best models
best_models_dict = dict()

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        wandb_api_key = os.environ['WANDB_API_KEY']
        wandb.login(key = wandb_api_key) # Enter your API key here
    except:
        print('Setting wandb usage to False')
        print('Add your wandb key in secrets so that you can use it')
        cfg.use_wandb = False

    def seed_everything(seed=cfg.seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    class Collate:
        """Data collator for training and improving efficiency"""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, batch):
            
            batch_len = max([len(sample["ids"]) for sample in batch])
            
            output = dict()
            output["ids"] = [sample["ids"] for sample in batch]
            output["mask"] = [sample["mask"] for sample in batch]
            output["targets"] = [sample["targets"] for sample in batch]
            
            if self.tokenizer.padding_side == "right":
                output["ids"] = [s + [self.tokenizer.pad_token_id] * (batch_len - len(s)) for s in output["ids"]]
                output["mask"] = [s + [0] * (batch_len - len(s)) for s in output["mask"]]
            else:
                output["ids"] = [[self.tokenizer.pad_token_id] * (batch_len - len(s)) + s for s in output["ids"]]
                output["mask"] = [[0] * (batch_len - len(s)) + s for s in output["mask"]]
                
                
            output["ids"] = torch.tensor(output["ids"], dtype = torch.long)
            output["mask"] = torch.tensor(output["mask"], dtype = torch.long)
            output["targets"] = torch.tensor(output["targets"], dtype = torch.float32)
            
            return output
        

    class Dataset(torch.utils.data.Dataset):
        """Pytorch dataset class for tokenizing the text and targets"""
        def __init__(self, texts, targets, tokenizer):
            self.texts = texts
            self.targets = targets
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            
            text = self.texts[idx]
            targets = self.targets[idx]
            encoding = self.tokenizer(text, add_special_tokens = True, max_length = cfg.max_len, padding = False, truncation = 'longest_first') 
            
            return {
                "ids": encoding["input_ids"], 
                "mask": encoding["attention_mask"],
                "targets": targets
            }
        
    class Model(nn.Module):
        """Model class"""
        def __init__(self, model_name, fold = None):
            super().__init__()

            self.model_name = model_name
            config = AutoConfig.from_pretrained(model_name)

            config.update(
                {
                    "output_hidden_states": True,
                    "hidden_dropout_prob": cfg.hidden_dropout_prob,
                    "attention_probs_dropout_prob" : cfg.hidden_dropout_prob,
                    "layer_norm_eps": cfg.layer_norm_eps,
                    "add_pooling_layer": False,
                    "num_labels": cfg.num_classes,
                }
            )
            
            self.config = config
            
            if cfg.use_gpl_checkpoint:
                print("Using GPL checkpoint")
                self.transformer = AutoModel.from_pretrained(f"{cfg.gpl_repo_id}-fold-{fold}", config=config)
            else:
                self.transformer = AutoModel.from_pretrained(model_name, config=config)
            
            if cfg.gradient_checkpointing_enable:
                self.transformer.gradient_checkpointing_enable()
            

            if cfg.pooling == "cls":
                self.pool = CLSPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, config = self.config)
            elif cfg.pooling == "mean":
                self.pool = MeanPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, config = self.config)
            elif cfg.pooling == "max":
                self.pool = MaxPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, config = self.config)
            elif cfg.pooling == "mean_max":
                self.pool = MeanMaxPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, config = self.config)
            elif cfg.pooling == "concat":
                self.pool = ConcatPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, pooling = cfg.concat_pooling, config = self.config)
            elif cfg.pooling == "lstm":
                self.pool = LSTMPooling(hidden_size = config.hidden_size, num_classes = cfg.num_classes, config = self.config)
            
            if cfg.freeze:
                self.freeze(cfg.start_freeze_layer, cfg.end_freeze_layer)
            
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
        def freeze(self, start_freeze = 0, end_freeze = 6):
            for i in range(start_freeze, end_freeze):
                for n,p in self.transformer.encoder.layer[i].named_parameters():
                    p.requires_grad = False
        
        def forward(self, ids, mask, targets = None):
            transformer_out = self.transformer(input_ids = ids, attention_mask = mask)
            logits = self.pool(transformer_out.last_hidden_state, mask, transformer_out.hidden_states)
            return logits
        
    class MCRMSELoss(nn.Module):
        """Custom MCRMSE loss function"""
        def __init__(self):
            super(MCRMSELoss, self).__init__()

        def forward(self, y_pred, y_true):
            colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
            return torch.mean(torch.sqrt(colwise_mse), dim=0)
        
    def criterion(inputs, targets):
        return nn.MSELoss()(inputs, targets)

    def criterion_mcrmse(inputs, targets):
        return MCRMSELoss()(inputs, targets)

    def get_optimizer_scheduler(model, num_train_steps):
        """get optimizer and scheduler"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
                "lr" : cfg.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr" : cfg.lr
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
                num_training_steps=num_train_steps,
                last_epoch=-1,
        )
        return optimizer, scheduler
    
    def get_grouped_parameters(model, num_train_steps):
        """get optimizer and scheduler"""
        ## Based on Gezi implementation
        # I think the layernorm weights and rel embeddings are tied to each layer
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "transformer" not in n],
                "weight_decay": 0.001,
                "lr" : cfg.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "transformer" not in n],
                "weight_decay": 0.0,
                "lr" : cfg.lr
            }
        ]
        
        lr = cfg.lr
        layers = list(model.transformer.encoder.layer)
        layers.reverse()
        for layer in layers:
            optimizer_params += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.001,
                    "lr" : lr
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr" : lr
                }
            ]
            lr *= cfg.llrd
        optimizer_params += [
                {
                    "params": [p for n, p in model.transformer.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.001,
                    "lr" : lr
                },
                {
                    "params": [p for n, p in model.transformer.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr" : lr
                }
            ]

        # There are three extra layers in the transformer model at the end
        last_three_layer_names = ["transformer.encoder.rel_embeddings.weight", "transformer.encoder.LayerNorm.weight","transformer.encoder.LayerNorm.bias"]
        optimizer_params += [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in last_three_layer_names)],
                "weight_decay": 0.001,
                "lr" : lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in last_three_layer_names)],
                "weight_decay": 0.0,
                "lr" : lr
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
                num_training_steps=num_train_steps,
                last_epoch=-1,
        )
        return optimizer, scheduler
    

    def train(epoch, model, train_loader, optimizer, scheduler, device, scaler, awp = None):
        """training pass"""
        model.train()
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(train_loader), total = len(train_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            with autocast():
                outputs = model(**batch)
                if cfg.criterion == "rdrop":
                    outputs2 = model(**batch)
                    loss = rdrop_loss(outputs, outputs2, batch["targets"], alpha = cfg.rdrop_alpha)
                if cfg.criterion == "rmse":
                    loss = criterion(outputs, batch["targets"])
                else:
                    loss = criterion_mcrmse(outputs, batch["targets"])

                if cfg.add_rank_loss:
                    loss += cfg.rank_loss_weight * rank_loss(outputs, batch["targets"])
            
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            
            losses.update(loss.item() * cfg.gradient_accumulation_steps , cfg.batch_size)
            scaler.scale(loss).backward()

            if cfg.use_awp:
                awp.attack_backward(batch, epoch)

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if cfg.use_wandb:
                wandb.log({
                    "train/loss": losses.val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": epoch * len(train_loader) + batch_idx,

                })
        
        return losses.avg

    @torch.inference_mode()
    def evaluate(epoch, model, valid_loader, device):
        """evaluate pass"""
        model.eval()
        all_targets = []
        all_outputs = []
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            outputs = model(**batch)
            if cfg.criterion == "rmse" or cfg.criterion == "rdrop":
                loss = criterion(outputs, batch["targets"])
            else:
                loss = criterion_mcrmse(outputs, batch["targets"])
            losses.update(loss.item(), cfg.batch_size)
            all_targets.extend(batch["targets"].detach().cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
        
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
        score = compute_mcrmse(all_outputs, all_targets)
        return score, losses.avg

    @torch.inference_mode()
    def predict(model, test_loader, device):
        """predict pass for calculating oof values"""
        model.eval()
        all_outputs = []

        for batch_idx, (batch) in tqdm(enumerate(test_loader), total = len(test_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            outputs = model(**batch)
            all_outputs.extend(outputs.cpu().numpy())
            
        all_outputs = np.vstack(all_outputs)
        return all_outputs


    def get_full_text(row, sep_token):
        if cfg.correct_spelling:
            columns = ["prompt_title","prompt_question", "corrected_text"]
        else:
            columns = ["prompt_title","prompt_question", "text"]
        texts = [row[col] for col in columns]
        if cfg.use_prompt_text:
            texts.append(row["prompt_text"])
        
        # Use specified columns
        if len(cfg.columns_to_use) > 0:
            columns = cfg.columns_to_use
            texts = [row[col] for col in columns]
        
        full_text = f" {sep_token} ".join(texts)
        row["full_text"] = full_text
        return row

    def main_fold(fold, seed, best_score):
        """Main loop"""
        # Seed everything
        seed_everything(seed=seed)
        if cfg.use_wandb:
            run = wandb.init(project=cfg.project_name, 
                            config=dict(cfg), 
                            group = cfg.model_name, 
                            reinit=True)
            wandb.define_metric("train/step")
            wandb.define_metric("valid/step")
            # define which metrics will be plotted against it
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("valid/*", step_metric="valid/step")
        
        pdf = pd.read_csv(cfg.train_prompt_file)
        sdf = pd.read_csv(cfg.train_summary_file)
        df = pdf.merge(sdf, on="prompt_id")
        # 4 prompt ids, 4 folds
        id2fold = {
            "39c16e": 0,
            "814d6b": 1,
            "3b9047": 2,
            "ebad26": 3,
        }
        df["fold"] = df["prompt_id"].map(id2fold)

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        sep_token = tokenizer.sep_token

        # Preparing the train texts and targets
    #     train_texts = train_df["text"].to_list()
    #     valid_texts = valid_df["text"].to_list()
        train_texts = train_df.progress_apply(get_full_text, args = (sep_token, ), axis = 1)["full_text"].to_list()
        valid_texts = valid_df.progress_apply(get_full_text, args = (sep_token, ), axis = 1)["full_text"].to_list()
        train_targets = train_df[list(cfg.target_columns)].values.tolist()
        valid_targets = valid_df[list(cfg.target_columns)].values.tolist()

        # Preparing the datasets and dataloaders
        collate_fn = Collate(tokenizer)
        train_ds = Dataset(train_texts, train_targets, tokenizer)
        valid_ds = Dataset(valid_texts, valid_targets, tokenizer)

        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size = cfg.batch_size, 
            shuffle = True, 
            collate_fn = collate_fn)

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size = cfg.batch_size,
            shuffle = False,
            collate_fn = collate_fn)
        
        # Preparing the model
        model = Model(cfg.model_name, fold)
        # Nosiy tune method for robust fine tuning
        if cfg.noisy_tune:
            model = noisy_tune(model)
        
        model = model.to(cfg.device)
        if cfg.compile:
            model = torch.compile(model)
        
        if cfg.use_wandb:
            wandb.watch(model)
        
        if cfg.multi_gpu:
            model = nn.DataParallel(model)
        
        num_train_steps = int(len(train_ds) / cfg.batch_size / cfg.gradient_accumulation_steps * cfg.epochs)

        if cfg.multi_gpu:
            if cfg.llrd < 1:
                optimizer, scheduler = get_grouped_parameters(model.module, num_train_steps)
            else:
                optimizer, scheduler = get_optimizer_scheduler(model.module, num_train_steps)
        else:
            if cfg.llrd < 1:
                print("Applying LLRD")
                optimizer, scheduler = get_grouped_parameters(model, num_train_steps)
            else:
                optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)

        scaler = GradScaler()
        global best_models_dict

        if cfg.use_awp:
            awp = AWP(model,
                optimizer,
                adv_lr=cfg.adv_lr,
                adv_eps=cfg.adv_eps,
                start_epoch=cfg.awp_start_epoch,
                loss_fn=criterion,
                scaler=scaler
            )
        else:
            awp = None
        # Training loop
        for epoch in range(cfg.epochs):
            print(f"FOLD : {fold}, EPOCH: {epoch + 1}")
            train_loss = train(epoch, model, train_loader, optimizer, scheduler, cfg.device, scaler, awp)
            valid_score, valid_loss = evaluate(epoch, model, valid_loader, cfg.device)
            print(f"\nValidation Metrics: {valid_score}")
            if cfg.use_wandb:
                wandb.log({"valid/train_loss_avg": train_loss, 
                        "valid/valid_loss_avg": valid_loss, 
                        "valid/mcrmse": valid_score["mcrmse"],
                        "valid/content_rmse": valid_score["content_rmse"],
                        "valid/wording_rmse": valid_score["wording_rmse"], 
                        "valid/step": epoch})
            
            if valid_score["mcrmse"] < best_score:
                best_score = valid_score["mcrmse"]
                if cfg.multi_gpu:
                    save_file_path = os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_fold{fold}_seed{cfg.seed}.bin")
                    torch.save(model.module.state_dict(), save_file_path)
                    best_models_dict[fold] = save_file_path
                else:
                    save_file_path = os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_fold{fold}_seed{cfg.seed}.bin")
                    torch.save(model.state_dict(), save_file_path)
                    best_models_dict[fold] = save_file_path
        
        if cfg.use_wandb:
            run.finish()
        
        torch.cuda.empty_cache()
        return best_score

    def train_whole_dataset():
        """Main loop for training on the whole dataset"""
        # Seed everything
        seed_everything(seed=cfg.seed)
        pdf = pd.read_csv(cfg.train_prompt_file)
        sdf = pd.read_csv(cfg.train_summary_file)
        df = pdf.merge(sdf, on="prompt_id")

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        sep_token = tokenizer.sep_token

        # Preparing the train texts and targets
    #     texts = df["text"].to_list()
        texts = df.apply(get_full_text, args = (sep_token,), axis = 1)["full_text"].to_list()
        targets = df[list(cfg.target_columns)].values.tolist()

        # Preparing the datasets and dataloaders
        collate_fn = Collate(tokenizer)
        train_ds = Dataset(texts, targets, tokenizer)

        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size = cfg.batch_size, 
            shuffle = True, 
            collate_fn = collate_fn)
        
        # Preparing the model
        model = Model(cfg.model_name)
        model = model.to(cfg.device)
        
        if cfg.multi_gpu:
            model = nn.DataParallel(model)
        
        num_train_steps = int(len(train_ds) / cfg.batch_size / cfg.gradient_accumulation_steps * cfg.epochs)

        if cfg.multi_gpu:
            if cfg.llrd < 1:
                optimizer, scheduler = get_grouped_parameters(model.module, num_train_steps)
            else:
                optimizer, scheduler = get_optimizer_scheduler(model.module, num_train_steps)
        else:
            if cfg.llrd < 1:
                optimizer, scheduler = get_grouped_parameters(model, num_train_steps)
            else:
                optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)

        scaler = GradScaler()
        # Training loop
        for epoch in range(cfg.epochs):
            train_loss = train(epoch, model, train_loader, optimizer, scheduler, cfg.device, scaler)
        
        if cfg.multi_gpu:
            save_file_path = os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_all_fold_seed{cfg.seed}.bin")
            torch.save(model.module.state_dict(), save_file_path)
        else:
            save_file_path = os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_all_fold_seed{cfg.seed}.bin")
            torch.save(model.state_dict(), save_file_path)
        torch.cuda.empty_cache()


    def calc_oof():
        pdf = pd.read_csv(cfg.train_prompt_file)
        sdf = pd.read_csv(cfg.train_summary_file)
        df = pdf.merge(sdf, on="prompt_id")
        

        # 4 prompt ids, 4 folds
        id2fold = {
            "39c16e": 0,
            "814d6b": 1,
            "3b9047": 2,
            "ebad26": 3,
        }
        df["fold"] = df["prompt_id"].map(id2fold)
        oof_df = df.copy()
        
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        sep_token = tokenizer.sep_token
        
        for fold in [0,1,2,3]:
            index = df[df["fold"] == fold].index
            test_df = df[df["fold"] == fold].reset_index(drop=True)
    #         test_texts = test_df["text"].to_list()
            test_texts = test_df.progress_apply(get_full_text, args = (sep_token, ), axis = 1)["full_text"].to_list()
            test_targets = test_df[list(cfg.target_columns)].values.tolist()
            
            # Preparing the model
            model = Model(cfg.model_name)
            model.load_state_dict(torch.load(best_models_dict[fold], map_location = "cpu"))
            model = model.to(cfg.device)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            
            if cfg.multi_gpu:
                model = nn.DataParallel(model)
            
            # Preparing the datasets and dataloaders
            collate_fn = Collate(tokenizer)
            test_ds = Dataset(test_texts, test_targets, tokenizer)
            test_loader = torch.utils.data.DataLoader(
                            test_ds,
                            batch_size = cfg.batch_size,
                            shuffle = False,
                            collate_fn = collate_fn,
                            drop_last = False)
            all_outputs = predict(model, test_loader, cfg.device)
            
            oof_df.loc[index, list(cfg.target_columns)] = all_outputs
        
        # Calculating the score on the oof df and then saving the oof df
        test_score = compute_mcrmse(oof_df[list(cfg.target_columns)].values, df[list(cfg.target_columns)].values)
        print(f"The oof score is {test_score}")
        columns_to_save = ['student_id','prompt_id', 'content', 'wording', 'fold']
        oof_df[columns_to_save].to_csv(os.path.join(cfg.output_dir, "oof.csv"), index = False)
        return test_score


    for fold in range(4):
        best_score = 1
        curr_best_score = main_fold(fold, cfg.seed, best_score)

    cv = float(calc_oof()['mcrmse'])

    cv = float(train_lgb(
        prompts_path = cfg.train_prompt_file,
        summaries_path = cfg.train_summary_file,
        model_name = cfg.model_name,
        oof_file_path = os.path.join(cfg.output_dir, "oof.csv")
    )["mcrmse"])

    print(f"CV after LGBM : {cv:.4f}")

    cfg.use_wandb = False
    if cfg.train_whole_dataset:
        train_whole_dataset()

    login(os.environ.get("HF_HUB_TOKEN"))

    api = HfApi()
    cfg.repo_id = f"jashdalvi/commonlit-kaggle-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}"
    # Creating a model repository in baseplate
    create_repo(cfg.repo_id, private= True, exist_ok=True)
    # Pushing the model to the hub
    api.upload_folder(
        folder_path = cfg.output_dir,
        path_in_repo = "/",
        repo_id = cfg.repo_id,
        repo_type = "model"
    )

    # Commenting out the kaggle api dataset upload code
    subprocess.run(["kaggle", "datasets", "init", "-p", cfg.output_dir], check=True)
    kaggle_dataset_metadata = {
    "title": f"commonlit-kaggle-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}",
    "id": f"jashdalvi99/commonlit-kaggle-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}".replace(".", ""),
    "licenses": [
        {
        "name": "CC0-1.0"
        }
    ]
    }
    # Overwriting the dataset metadata file
    with open(os.path.join(cfg.output_dir, "dataset-metadata.json"), "w") as f:
        json.dump(kaggle_dataset_metadata, f)
    # Uploading the dataset to kaggle
    subprocess.run(["kaggle", "datasets", "create", "-p", cfg.output_dir], check=True)

    # Deleting the output directory to save some space
    shutil.rmtree(cfg.output_dir)
    # Remove the local wandb dir to save some space
    shutil.rmtree("wandb")

if __name__ == "__main__":
    login(os.environ.get("HF_HUB_TOKEN"))
    main()



