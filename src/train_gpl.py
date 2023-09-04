import gpl
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import shutil
from dotenv import load_dotenv
from huggingface_hub import (
    login,
    HfApi,
    hf_hub_download,
    snapshot_download,
    create_repo,
)
import pandas as pd
import json

load_dotenv()

@hydra.main(config_path="config", config_name="config_gpl")
def main(cfg : DictConfig):
    login(os.environ.get("HF_HUB_TOKEN"))
    data_dir = cfg.data_dir

    for fold in [0,1,2,3]:
        
        ## Clearing the output path and creating a new one
        if os.path.exists(cfg.output_dir_path):
            shutil.rmtree(cfg.output_dir_path)

        if not os.path.exists(cfg.output_dir_path):
            os.makedirs(cfg.output_dir_path, exist_ok=True)
        
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        df = pd.read_csv(cfg.train_summary_file)
        id2fold = {
            "39c16e": 0,
            "814d6b": 1,
            "3b9047": 2,
            "ebad26": 3,
        }
        df["fold"] = df["prompt_id"].map(id2fold)

        train_df = df[df["fold"] != fold]

        id_count = 1
        with open(os.path.join(cfg.data_dir, "corpus.jsonl"), "w") as jsonl:
            for i, row in train_df.iterrows():
                line = {
                    '_id': str(id_count),
                    'title': "",
                    'text': row["text"].replace('\n', ' '),
                    'metadata': ""
                }
                id_count += 1
                jsonl.write(json.dumps(line)+'\n')

        # training the model with GPL
        gpl.train(
            path_to_generated_data=cfg.data_dir,
            base_ckpt=cfg.base_ckpt,
            # base_ckpt='GPL/msmarco-distilbert-margin-mse',
            # The starting checkpoint of the experiments in the paper
            gpl_score_function=cfg.score_function,
            rescale_range=cfg.rescale_range,
            # Note that GPL uses MarginMSE loss, which works with dot-product
            batch_size_gpl=cfg.batch_size,
            gpl_steps=cfg.train_steps,
            new_size=cfg.new_size,
            max_seq_length=cfg.max_seq_length,
            # Resize the corpus to `new_size` (|corpus|) if needed. When set to None (by default), the |corpus| will be the full size. When set to -1, the |corpus| will be set automatically: If QPP * |corpus| <= 250K, |corpus| will be the full size; else QPP will be set 3 and |corpus| will be set to 250K / 3
            queries_per_passage=cfg.queries_per_passage,
            # Number of Queries Per Passage (QPP) in the query generation step. When set to -1 (by default), the QPP will be chosen automatically: If QPP * |corpus| <= 250K, then QPP will be set to 250K / |corpus|; else QPP will be set 3 and |corpus| will be set to 250K / 3
            output_dir=cfg.output_dir_path,
            # evaluation_data=f"./{dataset}",
            # evaluation_output=f"evaluation/{dataset}",
            generator="BeIR/query-gen-msmarco-t5-base-v1",
            retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
            retriever_score_functions=["cos_sim", "cos_sim"],
            # Note that these two retriever model work with cosine-similarity
            cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
            qgen_prefix="qgen",
            # This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.
            do_evaluation=False,
            use_amp=True,  # One can use this flag for enabling the efficient float16 precision
        )
        api = HfApi()
        if cfg.push_to_hub:
            # Creating a model repository
            create_repo(f"{cfg.repo_id}-fold-{fold}", private=True, exist_ok=True)
            # Pushing the model to the hub
            api.upload_folder(
                folder_path=os.path.join(
                    cfg.output_dir_path, str(cfg.train_steps)
                ),
                path_in_repo="/",
                repo_id=cfg.repo_id,
                repo_type="model",
            )

if __name__ == "__main__":
    main()
    
