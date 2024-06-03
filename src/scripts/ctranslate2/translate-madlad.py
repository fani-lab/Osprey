# Translate only predatory conversations
import logging, sys, os
os.environ["CT2_VERBOSE"] = "2" # For logging purposes

import ctranslate2
import pandas as pd
from transformers import T5Tokenizer

import time
from glob import glob
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")
start_time_str = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
debug_file_handler = logging.FileHandler(f"logs/translation/{start_time_str}.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(FORMATTER_VERBOSE)
info_logger_file_path = f"logs/translation/{start_time_str}-info.log"
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

device = "cuda"

ct_model_path = "temp/madlad-3b/"
model_name = "jbochi/madlad400-3b-mt"
device="cuda"
logger.info("loading model")
translator = ctranslate2.Translator(ct_model_path, device)
logger.info("the model is loaded")
logger.info("loading the T5 tokenizer")
tokenizer = T5Tokenizer.from_pretrained(model_name)
logger.info("the tokenizer is loaded")

logger.info("reading the dataframes")
df = pd.read_csv("data/dataset-v2/train.csv", index_col=0)
logger.info("filtering the dataframe")
logger.info(f"before: {df.shape[0]}")
df["text"] = df["text"].fillna("")
df = df[df["predatory_conv"] == 1.0]
df = df[df['conv_size'] >= 6]
df.reset_index(inplace=True, drop=True)
df.sort_values("msg_word_count", inplace=True, ascending=False)
logger.info(f"after: {df.shape[0]}")

def decode(translation_results, tokenizer):
    return tokenizer.batch_decode([tokenizer.convert_tokens_to_ids(t.hypotheses[0]) for t in translation_results])

# def fw_bw_translate(texts, translator, tokenizer, target_langs, langs, batch_size=256):
#     texts = [tokenizer.convert_ids_to_tokens(t) for t in tokenizer.batch_encode_plus(texts).input_ids]
#     results = {}
#     for lang in target_langs:
#         fw_translations_subworded = translator.translate_batch([[langs[lang]] + t for t in texts], max_batch_size=batch_size)
#         bw_translations_subworded = translator.translate_batch([[langs['en']]+ tr.hypotheses[0] for tr in fw_translations_subworded], max_batch_size=batch_size)
#         results[lang] = (decode(fw_translations_subworded), decode(bw_translations_subworded))
#     return results


################
def fw_bw_translate(data, translator, tokenizer, target_langs, window_size=128):
    src_lang = "eng_Latn"
    l = len(data)
    fw_bw_results = {lang: ([None]*l, [None]*l) for lang in target_langs}
    target_langs_prefixes = {
        lang: [f"<2{lang}>"]  for lang in target_langs
    }
    source_prefix = [src_lang]

    counter = [0, 0]
    for s in range(0, l, window_size):
        counter[0] += 1
        
        beam_size = 1
        e = min(s+window_size, l)
        time_s = time.time()
        # source_sentences = [record.strip() for record in data[s:e]]
        source_sents_subworded = [tokenizer.convert_ids_to_tokens(t) for t in tokenizer.batch_encode_plus(data[s:e]).input_ids]
        # source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in sp.encode_as_pieces(source_sentences)]
        for lang in target_langs:
            # Forward translate
            fw_input = [target_langs_prefixes[lang] + ts for ts in source_sents_subworded]
            fw_translations_subworded = translator.translate_batch(fw_input, batch_type="tokens", max_batch_size=1024,
                                                                beam_size=beam_size)
            # Backward translate
            bw_input = [source_prefix + tr.hypotheses[0] for tr in fw_translations_subworded]
            bw_translations_subworded = translator.translate_batch(bw_input,
                batch_type="tokens", max_batch_size=1024, beam_size=beam_size)
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

try:
    logger.info("starting translation")
    target_langs = ['fr', 'zh']
    fw_bw_results = fw_bw_translate(df.iloc[:600]["text"].tolist(), translator, tokenizer, target_langs=target_langs, window_size=256)
    logger.info("translation finished")
    del translator, tokenizer
    logger.info("persisting as dataframes")
    new_df = df[["conv_id", "msg_line"]].copy()
    saved_paths = []
    for l, v in fw_bw_results.items():
        backward = v[1]
        forward = v[0]
        
        new_df['forward'] = pd.DataFrame(forward)
        new_df['backward'] = pd.DataFrame(backward)
        p = f"{ct_model_path}translation_fw_bw_{l}_predatory.csv"
        logger.info(f"saving at: {p}")
        new_df.to_csv(p, encoding="utf-32")
        saved_paths.append(p)

    logger.info("evaluating")
    rouge = evaluate.load('rouge')
    rs = None
    bw_df = None
    smoothing_function = SmoothingFunction().method2
    # just adding the metrics of backtranslation quality
    for p in saved_paths:
        logger.info(f"loading: {p}")

        bw_df = pd.read_csv(p, encoding="utf-32", index_col=0)
        hypotheses = bw_df["backward"].fillna("")
        references = df["text"].fillna("")
        rs = rouge.compute(predictions=hypotheses, references=references, use_aggregator=False)
        bleus = [0]*hypotheses.shape[0]
        for i in range(hypotheses.shape[0]):
            bleus[i] = sentence_bleu(references=[references.iloc[i].lower().split()], hypothesis=hypotheses.iloc[i].lower().split(), smoothing_function=smoothing_function)
        
        bw_df["bleus"] = pd.DataFrame(bleus)
        for r in rs.keys():
            bw_df[r] = pd.DataFrame(rs[r])
        
        logger.info(f"saving dataframe at: {p}")
        bw_df.to_csv(p, encoding="utf-32")
except Exception as e:
    logger.exception(e)
    raise e



"""

ace ace_Arab af am an ar ary arz as az ba ban bar be bg bho bjn bjn_Arab bm bn br bs bug ca
ceb crh_Latn cs cy da de din dv dz el en en_xx_simple eo es et eu fa fi fil fo fr fr_CA fr_ca
fur fuv fy ga gd gl gn gu ha he hi hne hr hu hy id ig io is it iu ja jv ka kk km kn ko kr kr_Arab
ks ks_Deva ku ky la lb li lij lmo lt ltg lv mag mg mi mk ml mn mni mr ms mt mwl my nb nds nds_NL
nds_nl ne nl nn no nus oc or pa pl prs ps pt pt_br ro ru rw sc scn sd se sh shn si simple sk sl so
sq sr sv sw szl ta taq taq_Tfng te tg th tk tl tr tt tzm ug uk ur uz vec vi wa wuu xh yi yo zh zh_Hant zh_cn zh_tw zu

"""