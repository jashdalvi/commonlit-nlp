import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

@st.cache_data
def load_prompts_df():
    prompts = pd.read_csv("../data/prompts_train.csv")
    return prompts

@st.cache_data
def load_summaries_df():
    summaries = pd.read_csv("../data/summaries_train.csv")
    return summaries

id2fold = {
    "39c16e": 0,
    "814d6b": 1,
    "3b9047": 2,
    "ebad26": 3,
}

prompts = load_prompts_df()
summaries = load_summaries_df()

count_bad_chars = 0
count_inv_commas = 0
content_score = 0
wording_score = 0
content_bad_score = 0
wording_bad_score = 0
for i, row in summaries.iterrows():
    if "¨" in row["text"]:
        wording_bad_score += row["wording"]
        content_bad_score += row["content"]
        count_bad_chars += 1
    
    if '"' in  row["text"]:
        wording_score += row["wording"]
        content_score += row["content"]
        count_inv_commas += 1

print(count_inv_commas)
print(content_score/count_inv_commas)
print(wording_score/count_inv_commas)

print(summaries["content"].mean())
print(summaries["wording"].mean())

print(count_bad_chars)
print(content_bad_score/count_bad_chars)
print(wording_bad_score/count_bad_chars)
    

st.title("Visualize the data")

row_number = st.number_input("Row Number", min_value=1, max_value=len(summaries), value=1, step=1)
example = summaries.iloc[row_number -1]
st.write("**Summary**")
st.text(example["text"])

st.write("**Content Score**")
content_score = example["content"]
st.write(content_score)

st.write("**Wording Score**")
wording_score = example["wording"]
st.write(wording_score)

st.write("**Prompt**")
prompt_id = example["prompt_id"]
prompt = prompts[prompts["prompt_id"] == prompt_id]

st.write("**Prompt question**")
st.text(prompt["prompt_question"].values[0])

st.write("**Prompt text**")
st.text(prompt["prompt_text"].values[0])

st.write("**Prompt title**")
st.text(prompt["prompt_title"].values[0])

# Replace chars = ["¨", "´", "[", "]"] 
# replace ' ' with "
# replace ’ with '




