import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import pickle

df = pd.read_csv("data/dataset-v2/train.csv")
df = df[['conv_id', 'msg_line', 'author_id',
       'time', 'msg_char_count', 'msg_word_count', 'conv_size', 'nauthor',
       'text', 'tagged_predator', 'predatory_conv']]
# I wanted to sort them, so sentences with the same number of tokens are next to each other in a batch. It helps, believe me.
df = df.sort_values("msg_word_count", ascending=False)
df["text"] = df["text"].fillna("")
max_length = min(int(df["msg_word_count"].max() * 1.4), 1024)
print(f"Max length of tokens: {max_length}")

# model specs and device
target_langs = ["fra_Latn", "zho_Hans"]
device = torch.device("cuda")
checkpoint = "facebook/nllb-200-distilled-600M"
# checkpoint = "facebook/nllb-200-3.3B"

torch.cuda.empty_cache()
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Tokenizing english tokens once to avoid tokenizing over and over again. (it just takes some memory, but who cares)
tokens = None
try:
    with open("temp/tokens.pkl", mode="rb") as f:
        tokens = pickle.load(f)
    tokens = {k: tokens[k] for k in tokens.keys()}
    print("loaded tokens from file")
except:
    print("no token file was found")

if not tokens:
    tokens = {"input_ids":[None]*df.shape[0], "attention_mask": [None]*df.shape[0]}
    window_size = 128
    for s in tqdm(range(0, df.shape[0], window_size), desc="tokenizing"):
        e = min(s+window_size, df.shape[0])
        temp = tokenizer(df.iloc[s:e]["text"].tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, return_length=max_length)
        tokens["input_ids"][s:e] = temp["input_ids"]
        tokens["attention_mask"][s:e] = temp["attention_mask"]

    tokens = {k: torch.stack(tokens[k]) for k in tokens.keys()}
    with open("temp/tokens.pkl", mode="wb") as f:
        pickle.dump(tokens, f)

# For storing the results of backward translation in the future. Simply just add your languages to `target_langs`
backward_token_ids = {lang: ([None]*df.shape[0], [None]*df.shape[0]) for lang in target_langs}

def backtranslate(tokens, target_langs, window_size=64):
    source_lang = "eng_Latn"
    l = len(tokens["input_ids"]) # Number of sentences for translation
    for s in tqdm(range(0, l, window_size), desc="translating"): # Batching based on window_size
        e = min(s+window_size, l)
        for lang in target_langs:
            forward = model.generate(input_ids=tokens["input_ids"][s:e].to(device), max_new_tokens=512,
                                    attention_mask=tokens["attention_mask"][s:e].to(device),
                                    forced_bos_token_id=tokenizer.lang_code_to_id[lang])
            # reuse the same gerented token ids and do not decode them.
            backward = model.generate(input_ids=forward, max_new_tokens=512,
                                    forced_bos_token_id=tokenizer.lang_code_to_id[source_lang])
            # append to repsective language list
            backward_token_ids[lang][0][s:e] = backward
            backward_token_ids[lang][1][s:e] = forward
    
    # decoding: you can simple decode and put them all in a list instead of `yield`ing.
    for l, (backward, forward) in backward_token_ids.items():
        yield (l, tokenizer.batch_decode(backward, skip_special_tokens=True),
               tokenizer.batch_decode(forward, skip_special_tokens=True))


torch.cuda.empty_cache()
for l, backward, forward in backtranslate(tokens, target_langs, window_size=32):
    new_df = df[["conv_id", "msg_line"]].copy()
    new_df['forward'] = pd.DataFrame(forward)
    new_df['backward'] = pd.DataFrame(backward)
    new_df.to_csv(f"translation_{l}.csv", encoding="utf-32")


