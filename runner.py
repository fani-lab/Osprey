import sys
import logging
import argparse

from src.main import RunTrainPipeline
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving, AuthorIDReplacer, AuthorIDReplacerBert
from src.utils.dataset import (BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset,
                               CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset, ConversationBagOfWords,
                               CNNConversationBagOfWords, ConversationBagOfWordsCleaned, SequentialConversationDataset,
                               ConversationBagOfWordsWithTriple, TemporalSequentialConversationOneHotDataset, TemporalAuthorsSequentialConversationOneHotDataset,
                               FineTuningDistilrobertaDataset, UncasedBaseBertEmbeddingDataset, UncasedBaseBertTokenizedDataset,
                               SequentialConversationBertBaseDataset, SequentialConversationEmbeddingDataset,
                               TemporalAuthorsSequentialConversationEmbeddingDataset, TemporalSequentialConversationBertBaseDataset,
                               TemporalAuthorsSequentialConversationOneHotDatasetFiltered, SequentialConversationDatasetFiltered,
                               TemporalSequentialConversationOneHotDatasetFiltered, TemporalAuthorsSequentialConversationBertBaseDataset,
                               TemporalSequentialConversationEmbeddingDataset, NAuthorsConversationBagOfWords, NAuthorTransformersEmbeddingDataset,
                               NAuthorTransformersBertDataset, Word2VecEmbeddingDataset, Word2VecFineTunedEmbeddingDataset,
                               SequentialWord2VecEmbeddingDataset, NAuthorWord2VecEmbeddingDataset, NAuthorFinetunedWord2VecEmbeddingDataset,
                               TemporalAuthorsSequentialConversationWord2VecDataset, TemporalAuthorsSequentialConversationFinetunedWord2VecDataset,
                               TemporalAuthorsSequentialConversationDistilrobertaPretainedDataset, SequentialWord2VecFinetunedDataset,
                               TransformersDistilrobertaFinedtunedDataset, SequentialConversationDistilrobertaFinetunedDataset,
                               NAuthorTransformersDistilrobertaMoreTrainedDataset)
from src.utils.loss_functions import WeightedBinaryCrossEntropy, DynamicSuperLoss
from src.models import ANNModule, EbrahimiCNN, BaseRnnModule, LSTMModule, GRUModule, SuperDynamicLossANN, DistilrobertaFinetuningClassifier, BaseSingleVectorMachine
from src.mappings import register_mappings, register_mappings_torch, register_command, COMMANDS
import settings
from src.scripts import (CreateConversations, BalanceDatasetsForVersionTwo, CreateConversationToySet,
                            BalanceSequentialDatasetsForVersionTwo, PrintMappings, XML2CSV, finetune_tranformer_per_message,
                            MergeTranslationWithOriginal, StandardizeBacktranslation)
from src.utils.dataset import SequentialConversationDataset


def init_logger():
    FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
    FORMATTER_VERBOSE = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

    debug_file_handler = logging.FileHandler(f"logs/{settings.get_start_time()}.log")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(FORMATTER_VERBOSE)
    info_logger_file_path = f"logs/{settings.get_start_time()}-info.log"
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


def init_parser():
    main_parser = argparse.ArgumentParser(prog="osprey", description="running", add_help=False)
    main_parser.add_argument('-h', '--help',
        action='help', default=argparse.SUPPRESS,
        help='show this help message and exit')

    main_parser.add_argument("--log", "-l", action="store_true", default=False, help="apply logging for most of commands")
    
    subparsers = main_parser.add_subparsers()
    for cmd, cls in COMMANDS.items():
        obj = cls()
        callback, args = obj.get_actions_and_args()
        parser = subparsers.add_parser(cmd, help=obj.help(), add_help=True)
        for arg in args:
            flags = arg.pop("flags", list())
            flags = flags if isinstance(flags, list) or isinstance(flags, tuple) else (flags,)
            parser.add_argument(*flags, **arg)
        parser.set_defaults(func=callback)
    

    return main_parser

if __name__ == "__main__":

    register_command(PrintMappings)
    register_command(RunTrainPipeline)
    register_command(CreateConversations)
    register_command(BalanceDatasetsForVersionTwo)
    register_command(CreateConversationToySet)
    register_command(BalanceSequentialDatasetsForVersionTwo)
    register_command(XML2CSV)
    register_command(StandardizeBacktranslation)
    register_command(MergeTranslationWithOriginal)

    register_mappings_torch()

    register_mappings(DynamicSuperLoss)

    register_mappings(NLTKStopWordRemoving)
    register_mappings(PunctuationRemoving)
    register_mappings(RepetitionRemoving)
    register_mappings(AuthorIDReplacer)
    register_mappings(AuthorIDReplacerBert)

    register_mappings(BagOfWordsDataset)
    register_mappings(TimeBasedBagOfWordsDataset)
    register_mappings(TransformersEmbeddingDataset)
    register_mappings(TransformersDistilrobertaFinedtunedDataset)
    register_mappings(UncasedBaseBertEmbeddingDataset)
    register_mappings(CaseSensitiveBertEmbeddingDataset)
    register_mappings(GloveEmbeddingDataset)
    register_mappings(ConversationBagOfWords)
    register_mappings(CNNConversationBagOfWords)
    register_mappings(ConversationBagOfWordsCleaned)
    register_mappings(SequentialConversationDataset)
    register_mappings(ConversationBagOfWordsWithTriple)
    register_mappings(TemporalSequentialConversationOneHotDataset)
    register_mappings(TemporalAuthorsSequentialConversationOneHotDataset)
    register_mappings(TemporalSequentialConversationEmbeddingDataset)
    register_mappings(TemporalAuthorsSequentialConversationEmbeddingDataset)
    register_mappings(TemporalSequentialConversationBertBaseDataset)
    register_mappings(TemporalAuthorsSequentialConversationBertBaseDataset)
    register_mappings(TemporalAuthorsSequentialConversationOneHotDatasetFiltered)
    register_mappings(SequentialConversationDatasetFiltered)
    register_mappings(TemporalSequentialConversationOneHotDatasetFiltered)
    register_mappings(FineTuningDistilrobertaDataset)
    register_mappings(UncasedBaseBertTokenizedDataset)
    register_mappings(SequentialConversationBertBaseDataset)
    register_mappings(SequentialConversationEmbeddingDataset)
    register_mappings(TemporalAuthorsSequentialConversationDistilrobertaPretainedDataset)
    register_mappings(SequentialWord2VecEmbeddingDataset)
    register_mappings(SequentialWord2VecFinetunedDataset)
    register_mappings(SequentialConversationDistilrobertaFinetunedDataset)
    register_mappings(TemporalAuthorsSequentialConversationWord2VecDataset)
    register_mappings(TemporalAuthorsSequentialConversationFinetunedWord2VecDataset)
    register_mappings(NAuthorsConversationBagOfWords)
    register_mappings(NAuthorTransformersDistilrobertaMoreTrainedDataset)
    register_mappings(NAuthorTransformersEmbeddingDataset)
    register_mappings(NAuthorTransformersBertDataset)
    register_mappings(NAuthorWord2VecEmbeddingDataset)
    register_mappings(NAuthorFinetunedWord2VecEmbeddingDataset)
    register_mappings(Word2VecEmbeddingDataset)
    register_mappings(Word2VecFineTunedEmbeddingDataset)

    register_mappings(WeightedBinaryCrossEntropy)

    register_mappings(ANNModule)
    register_mappings(EbrahimiCNN)
    register_mappings(BaseRnnModule)
    register_mappings(LSTMModule)
    register_mappings(GRUModule)
    register_mappings(SuperDynamicLossANN)
    register_mappings(DistilrobertaFinetuningClassifier)
    register_mappings(BaseSingleVectorMachine)

    parser = init_parser()

    args = parser.parse_args()
    
    kwargs = vars(args)
    do_logging = kwargs.pop("log")
    func = kwargs.pop("func")
    if do_logging:
        logger = init_logger()

    func(**kwargs)
