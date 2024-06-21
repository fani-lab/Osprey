import logging, sys, os
os.environ['BATCH_SIZE'] = "512"
import ctranslate2
import sentencepiece as spm
import pandas as pd
from tqdm import tqdm

from time import time, strftime, localtime
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

from src.utils.commons import calculate_embedding_similarity, get_token_count_diff

# ct_model_path = "temp/facebook/nllb-200-distilled-600M/"
ct_model_path = "temp/facebook/nllb-200-3.3B/"
sp_model_path = "flores200sacrebleuspm"
start_time_str = strftime("%m-%d-%Y-%H-%M-%S", localtime())

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")
log_root_path = f"logs/translation/{start_time_str}"
os.makedirs(os.path.dirname(log_root_path), exist_ok=True)
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


device = "cuda"
logger.info("loading the sentence piece processor")
# Load the source SentecePiece model
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)
logger.info("the sentence piece processor is loaded")

logger.info("loading model")
translator = ctranslate2.Translator(ct_model_path, device)
logger.info("model is loaded")
logger.info("reading the dataframes")
df = pd.read_csv("data/dataset-v2/train.csv", index_col=0)
logger.info("filtering the dataframe")
df["text"] = df["text"].fillna("")
logger.info("no partitioning")
df = df[df["predatory_conv"] == 1.0]
df = df[df['conv_size'] >= 6]
df.reset_index(inplace=True, drop=True)

logger.info(f"before: {df.shape[0]}")
# df = df[(df["text"] != '')]
logger.info(f"after: {df.shape[0]}")


def decode(subworded, lang):
    subworded = [translation.hypotheses[0][1:] if lang == translation.hypotheses[0][0] else translation.hypotheses[0] for translation in subworded]
    return sp.decode(subworded)

def fw_bw_translate(data, target_langs, window_size=128, beam_size=2):
    src_lang = "eng_Latn"
    l = len(data)
    fw_bw_results = {lang: ([None]*l, [None]*l) for lang in target_langs}
    target_langs_prefixes = {
        lang: [[lang]] * window_size for lang in target_langs
    }
    source_prefix = [[src_lang]] * window_size
    
    counter = [0, 0]
    for s in range(0, l, window_size):
        counter[0] += 1
        
        e = min(s+window_size, l)
        time_s = time()
        source_sentences = [record.strip() for record in data[s:e]]
        source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in sp.encode_as_pieces(source_sentences)]
        for lang in target_langs:
            # Forward translate
            fw_translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=512,
                                                                beam_size=beam_size, target_prefix=target_langs_prefixes[lang][0:len(source_sents_subworded)])
            # Backward translate
            bw_translations_subworded = translator.translate_batch([tr.hypotheses[0] + ["</s>",] for tr in fw_translations_subworded], # important
                batch_type="tokens", max_batch_size=512, beam_size=beam_size, target_prefix=source_prefix[0:len(source_sents_subworded)])
            # make tokens into sentences
            bw = decode(bw_translations_subworded, src_lang)
            fw = decode(fw_translations_subworded, lang)
            # persist generated tokens
            fw_bw_results[lang][0][s:e] = bw
            fw_bw_results[lang][1][s:e] = fw
            tt = time()-time_s
            counter[1] += tt
        logger.info(f"from: {s} to {e} out of {l} ({(e/l * 100):>0.5f}%) at {tt} | avg {counter[1]/counter[0]} time of each iteration")
    return fw_bw_results

try:
    logger.info("starting translation")
    # target_langs = ["deu_Latn", "pes_Arab", "tur_Latn", "spa_Latn"]
    # target_langs = ["isl_Latn"]#, "deu_Latn"]
    # target_langs = ["cat_Latn"]#, "fra_Latn"]
    # target_langs = ['mya_Mymr']#, 'zho_Hans']
    # target_langs = ['hin_Deva']#, 'pes_Arab']
    logger.info(f'target languages: ' + ' - '.join(target_langs))
    logger.info(f"Tokenizer and model: {ct_model_path}")
    fw_bw_results = fw_bw_translate(df["text"].tolist(), target_langs, window_size=512, beam_size=2)
    logger.info("translation finished")
    del translator
    logger.info("persisting as dataframes")
    new_df = df[["conv_id", "msg_line"]].copy()
    saved_paths = []
    for l, v in fw_bw_results.items():
        backward = v[0]
        forward = v[1]
        
        new_df['forward'] = pd.DataFrame(forward)
        new_df['backward'] = pd.DataFrame(backward)
        p = f"{ct_model_path}nllb_200_3.3B-{l}.csv"
        logger.info(f"saving at: {p}")
        new_df.to_csv(p, encoding="utf-32")
    saved_paths.append(p)

    logger.info("evaluating")
    rouge = evaluate.load('rouge')
    rs = None
    bw_df = None
    smoothing_function = SmoothingFunction().method2

    for p in saved_paths:
        logger.info(f"loading: {p}")
        bw_df = pd.read_csv(p, encoding="utf-32", index_col=0)
        hypotheses = bw_df["backward"].fillna("").tolist()
        references = df["text"].fillna('').tolist()
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
        bw_df = pd.merge(df, bw_df, on=["conv_id", "msg_line"])
        bw_df.to_csv(p, encoding="utf-8")

except Exception as e:
    logger.exception(e)
    raise e
