from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy
import re
from spellchecker import SpellChecker
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import textdistance
import textstat

class Preprocessor:
    def __init__(self, 
                model_name: str,
                ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.STOP_WORDS = set(stopwords.words('english'))
        
        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        self.speller = SpellChecker() #Speller(lang='en')
        
    def count_text_length(self, df: pd.DataFrame, col:str) -> pd.Series:
        """ text length """
        tokenizer=self.tokenizer
        return df[col].progress_apply(lambda x: len(tokenizer.encode(x)))

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """        
        def check_is_stop_word(word):
            return word in self.STOP_WORDS
        
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))
            
    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int):
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        # # Optionally, you can get the frequency of common n-grams for a more nuanced analysis
        # original_ngram_freq = Counter(ngrams(original_words, n))
        # summary_ngram_freq = Counter(ngrams(summary_words, n))
        # common_ngram_freq = {ngram: min(original_ngram_freq[ngram], summary_ngram_freq[ngram]) for ngram in common_ngrams}

        return len(common_ngrams)
    
    def ner_overlap_count(self, row, mode:str):
        model = self.spacy_ner_model
        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])
        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)
        
        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))
        
        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}

    
    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary)>0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        
        wordlist=text.split()
        amount_miss = len(list(self.speller.unknown(wordlist)))

        return amount_miss

    def get_stopwords_rel(self,  text):
        text_words = word_tokenize(text)
        num_stopwords = sum([word in self.STOP_WORDS for word in text_words])
        
        return num_stopwords/len(text_words)

    def get_stat_features(self, df, text_col="text"):
        df["num_unique_words"] = df[text_col].apply(lambda x: len(set(x.split())))
        df["num_words"] = df[text_col].apply(lambda x: len(x.split()))
        df["num_sentences"] = df[text_col].apply(lambda x: len(x.split('.')))
        df["isupper"] = df[text_col].apply(lambda x: x[0].isupper())
        df["mean_num_words"] = df[text_col].apply(lambda x: np.mean([len(e.split()) for e in x.split('.')]))
        df["mean_num_unique_words"] = df[text_col].apply(lambda x: np.mean([len(set(e.split())) for e in x.split('.')]))
        df["num_slash"] = df[text_col].apply(lambda x: x.count("\n"))
        df["paragraph_count"] = df[text_col].apply(lambda x: x.count("\n\n"))
        df["upper_count"] = df[text_col].apply(lambda x: np.sum([w.isupper() for w in x.split()])/len(x.split()))
        df["syntax_count"] = df[text_col].apply(lambda x: x.count(",") + x.count("-") + x.count(";") + x.count(":"))
        df["is_end_with_dot"] = df[text_col].apply(lambda x: int(x[-1] == "."))
        
    #     df["num_unique_words_prompt_text"] = df["prompt_text"].apply(lambda x: len(set(x.split()))) - df["num_unique_words"]
    #     df["num_words_prompt_text"] = df["prompt_text"].apply(lambda x: len(x.split())) - df["num_words"]
    #     df["num_sentences_prompt_text"] = df["prompt_text"].apply(lambda x: len(x.split('.'))) - df["num_sentences"]

        # df["lcsstr"] = df.progress_apply(lambda row: textdistance.lcsstr(row.text, row.prompt_text).split().__len__(), axis=1)
        
        df["stopwords_rel"] = df[text_col].progress_apply(lambda x: self.get_stopwords_rel(x))
    #     df['diff_emb_sb'] = df.progress_apply(lambda x: np.sum(get_emb_sb(x["text"])*get_emb_sb(x["prompt_text"])), axis=1)
    #     df['diff_emb_qa'] = df.progress_apply(lambda x: np.sum(get_emb_qa(x["text"])*get_emb_qa(x["prompt_question"])), axis=1)
    #     df['ch'] = df.progress_apply(lambda x: get_ch_label(x["text"]), axis=1)
    #     df['rw'] = df.progress_apply(lambda x: get_rw(x["prompt_question"], x["text"]), axis=1)

        df['automated_readability_index'] = df[text_col].progress_apply(lambda x: textstat.automated_readability_index(x))
        df['coleman_liau_index'] = df[text_col].progress_apply(lambda x: textstat.coleman_liau_index(x))
        df['smog_index'] = df[text_col].progress_apply(lambda x: textstat.smog_index(x))
        df['dale_chall_readability_score'] = df[text_col].progress_apply(lambda x: textstat.dale_chall_readability_score(x))
        df['linsear_write_formula'] = df[text_col].progress_apply(lambda x: textstat.linsear_write_formula(x))
        df['gunning_fog'] = df[text_col].progress_apply(lambda x: textstat.gunning_fog(x))
        df['text_standard_float'] = df[text_col].progress_apply(lambda x: textstat.text_standard(x, float_output=True))
        df['spache_readability'] = df[text_col].progress_apply(lambda x: textstat.spache_readability(x))
        df['rix'] = df[text_col].progress_apply(lambda x: textstat.rix(x))
        df['lix'] = df[text_col].progress_apply(lambda x: textstat.lix(x))

        return df
    
    def run(self, 
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:
        
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )

        )
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1 
        )
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        
        # Crate dataframe with count of each category NERs overlap for all the summaries
        # Because it spends too much time for this feature, I don't use this time.
#         ners_count_df  = input_df.progress_apply(
#             lambda row: pd.Series(self.ner_overlap_count(row, mode=mode), dtype='float64'), axis=1
#         ).fillna(0)
#         self.ner_keys = ners_count_df.columns
#         ners_count_df['sum'] = ners_count_df.sum(axis=1)
#         ners_count_df.columns = ['NER_' + col for col in ners_count_df.columns]
#         # join ner count dataframe with train dataframe
#         input_df = pd.concat([input_df, ners_count_df], axis=1)
        
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        input_df = self.get_stat_features(input_df)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])