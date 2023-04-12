# This file is for sake of compatibility between different development environment
from src.main import run
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving
from src.utils.dataset import (BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset,
                               CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset, ConversationBagOfWords,
                               CNNConversationBagOfWords)
from src.utils.loss_functions import WeightedBinaryCrossEntropy
from src.models import ANNModule, RnnModule, EbrahimiCNN
from settings.mappings import register_mappings, register_mappings_torch
from src.scripts import create_conversations, balance_datasets_for_version_two


if __name__ == "__main__":
    # create_conversations()
    register_mappings_torch()

    register_mappings(NLTKStopWordRemoving)
    register_mappings(PunctuationRemoving)
    register_mappings(RepetitionRemoving)

    register_mappings(BagOfWordsDataset)
    register_mappings(TimeBasedBagOfWordsDataset)
    register_mappings(TransformersEmbeddingDataset)
    register_mappings(CaseSensitiveBertEmbeddingDataset)
    register_mappings(GloveEmbeddingDataset)
    register_mappings(ConversationBagOfWords)
    register_mappings(CNNConversationBagOfWords)

    register_mappings(WeightedBinaryCrossEntropy)

    register_mappings(ANNModule)
    register_mappings(RnnModule)
    register_mappings(EbrahimiCNN)

    run()
    # balance_datasets_for_version_two()
