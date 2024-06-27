# Translate only predatory conversations
import logging, sys, os
# os.environ["CT2_VERBOSE"] = "2" # For logging purposes
os.environ['BATCH_SIZE'] = "512"
import ctranslate2
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import time
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from src.utils.commons import calculate_embedding_similarity, get_token_count_diff

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")
start_time_str = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
log_root_path = f"logs/translation/{start_time_str}"
debug_file_handler = logging.FileHandler(f"{log_root_path}.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(FORMATTER_VERBOSE)
info_logger_file_path = f"{log_root_path}-info.log"
info_file_handler = logging.FileHandler(info_logger_file_path)
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(FORMATTER_VERBOSE)

info_terminal_handler = logging.StreamHandler(sys.stdout)
info_terminal_handler.setLevel(logging.INFO)
info_terminal_handler.setFormatter(FORMATTER)

logger = logging.getLogger()
logger.addHandler(debug_file_handler)
logger.addHandler(info_file_handler)
logger.addHandler(info_terminal_handler)
logger.setLevel(logging.DEBUG)
logger.info(f"info-level logger file handler created at: {info_logger_file_path}")


ct_model_path = "temp/m2m100_1.2B_f16/"
model_name = "facebook/m2m100_1.2B"

# ct_model_path = "temp/m2m100_418M/"
# model_name = "facebook/m2m100_418M"
device="cuda"
logger.info("loading model")
translator = ctranslate2.Translator(ct_model_path, device=device)
logger.info("the model is loaded")
logger.info("loading the tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = "en"
logger.info("the tokenizer is loaded")

logger.info("reading the dataframes")
original_df = pd.read_csv("data/dataset-v2/train.csv", index_col=0)
logger.info("filtering the dataframe")
logger.info(f"before: {original_df.shape[0]}")
original_df["text"] = original_df["text"].fillna("")
df = original_df[original_df["predatory_conv"] == 1.0]
df = df[df['conv_size'] >= 6]
df.reset_index(inplace=True, drop=True)
df.sort_values("msg_word_count", inplace=True, ascending=False)
logger.info(f"after: {df.shape[0]}")


def decode(translation_results, tokenizer):
    return tokenizer.batch_decode([tokenizer.convert_tokens_to_ids(t.hypotheses[0][1:]) for t in translation_results])

################
def fw_bw_translate(data, translator, tokenizer, target_langs, window_size=128, beam_size=2):
    src_lang = "en"
    l = len(data)
    fw_bw_results = {lang: ([None]*l, [None]*l) for lang in target_langs}
    target_langs_prefixes = {
        lang: [[tokenizer.lang_code_to_token[lang]]]*window_size for lang in target_langs
    }
    source_prefix = [[tokenizer.lang_code_to_token[src_lang]]]*window_size
    counter = [0, 0]
    for s in tqdm(range(0, l, window_size)):
        counter[0] += 1
        e = min(s+window_size, l)
        time_s = time.time()
        source_sents_subworded = [tokenizer.convert_ids_to_tokens(t) for t in tokenizer.batch_encode_plus(data[s:e]).input_ids]
        for lang in target_langs:
            # Forward translate
            fw_translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=1024, beam_size=beam_size, target_prefix=target_langs_prefixes[lang][0:len(source_sents_subworded)])
            # Backward translate
            if fw_translations_subworded[0].hypotheses[0][0] != tokenizer.lang_code_to_token[lang]:
                raise ValueError()
            bw_input = [tr.hypotheses[0] + ['</s>'] for tr in fw_translations_subworded]
            bw_translations_subworded = translator.translate_batch(bw_input, batch_type="tokens", max_batch_size=1024, beam_size=beam_size, target_prefix=source_prefix[0:len(source_sents_subworded)])
            # make tokens into sentences
            bw = decode(bw_translations_subworded, tokenizer)
            fw = decode(fw_translations_subworded, tokenizer)
            # persist generated tokens
            fw_bw_results[lang][0][s:e] = bw
            fw_bw_results[lang][1][s:e] = fw
            tt = time.time()-time_s
            counter[1] += tt
            
        logger.info(f"from: {s} to {e} out of {l} ({(e/l * 100):>0.5f}%) at {tt} | avg {counter[1]/counter[0]} time of each iteration")
    return fw_bw_results

##########################

def shit(fw_bw, data, lang, a):
    print(data[a], end='\n\n')
    print(fw_bw[lang][1][a], end='\n\n')
    print(fw_bw[lang][0][a], end='\n\n')

try:
    logger.info("starting translation")
    # target_langs = ['is', 'de'] # 0
    # target_langs = ['ca', 'fr'] # 1
    # target_langs = ['my', 'zh'] # 2
    # target_langs = ['fa', 'hi'] # 3
    # target_langs = ['fr']       # 4
    logger.info(f'target languages: ' + ' - '.join(target_langs))
    logger.info(f"Tokenizer and model: {ct_model_path}")
    fw_bw_results = fw_bw_translate(df["text"].tolist(), translator, tokenizer, target_langs=target_langs, window_size=512, beam_size=2)
    logger.info("translation finished")
    del translator, tokenizer
    logger.info("persisting as dataframes")
    new_df = df[["conv_id", "msg_line"]].copy().reset_index()
    saved_paths = []
    for l, v in fw_bw_results.items():
        backward = v[0]
        forward = v[1]
        
        new_df['forward'] = pd.DataFrame(forward)
        new_df['backward'] = pd.DataFrame(backward)
        p = f"{ct_model_path}m2m100_1.2B_f16{l}_noformat.csv"
        logger.info(f"saving at: {p}")
        new_df.to_csv(p, encoding="utf-8")
        saved_paths.append(p)

    logger.info("evaluating")
    rouge = evaluate.load('rouge')
    rs = None
    bw_df = None
    smoothing_function = SmoothingFunction().method2
        # just adding the metrics of backtranslation quality
    for p in saved_paths:
        logger.info(f"loading: {p}")
        bw_df = pd.read_csv(p, encoding="utf-8", index_col=0)
        hypotheses = bw_df["backward"].fillna("").tolist()
        references = df["text"].fillna("").tolist()
        rs = rouge.compute(predictions=hypotheses, references=references, use_aggregator=False)
        bleus = [0]*len(hypotheses)
        for i in range(len(hypotheses)):
            bleus[i] = sentence_bleu(references=[references[i].lower().split()], hypothesis=hypotheses[i].lower().split(), smoothing_function=smoothing_function)
        bw_df["bleus"] = pd.DataFrame(bleus)
        for r in rs.keys():
            bw_df[r] = pd.DataFrame(rs[r])
        # semsim
        bw_df['semsim-declutr'] = pd.DataFrame(calculate_embedding_similarity(references, hypotheses, "johngiorgi/declutr-base"))
        bw_df['semsim-minilm'] = pd.DataFrame(calculate_embedding_similarity(references, hypotheses, "sentence-transformers/all-MiniLM-L6-v2"))
        bw_df['token_count_dif'] = get_token_count_diff(references, hypotheses)
        logger.info(f"saving dataframe at: {p}")
        bw_df = pd.merge(original_df, bw_df, on=["conv_id", "msg_line"])
        bw_df.to_csv(p, encoding="utf-8")

except Exception as e:
    logger.exception(e)
    raise e
