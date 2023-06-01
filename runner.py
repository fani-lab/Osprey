# This file is for sake of compatibility between different development environment
from src.main import run
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving, AuthorIDReplacer
from src.utils.dataset import (BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset,
                               CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset, ConversationBagOfWords,
                               CNNConversationBagOfWords, ConversationBagOfWordsCleaned, SequentialConversationDataset,
                               ConversationBagOfWordsWithTriple, TemporalSequentialConversationDataset)
from src.utils.loss_functions import WeightedBinaryCrossEntropy, DynamicSuperLoss
from src.models import ANNModule, EbrahimiCNN, BaseRnnModule, LSTMModule, GRUModule, SuperDynamicLossANN
from settings.mappings import register_mappings, register_mappings_torch
from src.scripts import (create_conversations, balance_datasets_for_version_two, create_conversation_toy_set, generate_stats,
                         balance_sequential_datasets_for_version_two)
from src.utils.dataset import SequentialConversationDataset

if __name__ == "__main__":
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
    register_mappings(TemporalSequentialConversationDataset)

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
