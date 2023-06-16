# This file is for sake of compatibility between different development environment
import sys
import time
import logging

from src.main import run
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving, AuthorIDReplacer
from src.utils.dataset import (BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset,
                               CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset, ConversationBagOfWords,
                               CNNConversationBagOfWords, ConversationBagOfWordsCleaned, SequentialConversationDataset,
                               ConversationBagOfWordsWithTriple, TemporalSequentialConversationOneHotDataset, TemporalAuthorsSequentialConversationOneHotDataset,
                               FineTuningBertDataset)
from src.utils.loss_functions import WeightedBinaryCrossEntropy, DynamicSuperLoss
from src.models import ANNModule, EbrahimiCNN, BaseRnnModule, LSTMModule, GRUModule, SuperDynamicLossANN
from settings.mappings import register_mappings, register_mappings_torch
from src.scripts import (create_conversations, balance_datasets_for_version_two, create_conversation_toy_set, generate_stats,
                         balance_sequential_datasets_for_version_two, finetune_tranformer_per_message)
from src.utils.dataset import SequentialConversationDataset

START_TIME = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

def init_logger():
    FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
    FORMATTER_VERBOSE = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

    debug_file_handler = logging.FileHandler(f"logs/{START_TIME}.log")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(FORMATTER_VERBOSE)
    info_logger_file_path = f"logs/{START_TIME}-info.log"
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

    return logger


if __name__ == "__main__":
    logger = init_logger()
    # create_conversations()
    register_mappings_torch()

    register_mappings(DynamicSuperLoss)

    register_mappings(NLTKStopWordRemoving)
    register_mappings(PunctuationRemoving)
    register_mappings(RepetitionRemoving)
    register_mappings(AuthorIDReplacer)

    register_mappings(BagOfWordsDataset)
    register_mappings(TimeBasedBagOfWordsDataset)
    register_mappings(TransformersEmbeddingDataset)
    register_mappings(CaseSensitiveBertEmbeddingDataset)
    register_mappings(GloveEmbeddingDataset)
    register_mappings(ConversationBagOfWords)
    register_mappings(CNNConversationBagOfWords)
    register_mappings(ConversationBagOfWordsCleaned)
    register_mappings(SequentialConversationDataset)
    register_mappings(ConversationBagOfWordsWithTriple)
    register_mappings(TemporalSequentialConversationOneHotDataset)
    register_mappings(TemporalAuthorsSequentialConversationOneHotDataset)
    register_mappings(FineTuningBertDataset)

    register_mappings(WeightedBinaryCrossEntropy)

    register_mappings(ANNModule)
    register_mappings(EbrahimiCNN)
    register_mappings(BaseRnnModule)
    register_mappings(LSTMModule)
    register_mappings(GRUModule)
    register_mappings(SuperDynamicLossANN)

    run()
    # balance_datasets_for_version_two()
    # create_conversation_toy_set()
    # generate_stats()
    # finetune_tranformer_per_message("cuda", dataset_name="finetuning-v2-dataset", model_output_path=None)
