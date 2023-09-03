import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def run():
    fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base", device = 0)
    df = pd.read_csv("../data/summaries_train.csv")
    texts = df["text"].to_list()
    corrected_text = []
    df["corrected_text"] = ""
    dataset = Dataset(texts)
    batch_size = 64
    for out in tqdm(fix_spelling(dataset, batch_size=batch_size, max_length = 2048), total=len(dataset)):
        corrected_text.append(out[0]["generated_text"])
    # # Updating the text column by corrected text
    for i, _ in df.iterrows():
        df.loc[i, "corrected_text"] = corrected_text[i]
    df.to_csv("../data/summaries_train.csv", index = False)

if __name__ == "__main__":
    run()