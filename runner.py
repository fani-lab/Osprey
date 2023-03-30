# This file is for sake of compatibility between different development environment
from src.main import run
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving
from src.utils.dataset import BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset, CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset
from src.utils.commons import message_csv2conversation_csv, force_open
from src.models import ANNModule, RnnModule
from settings.mappings import register_mappings, register_mappings_torch


def create_conversations():
    df = message_csv2conversation_csv("data/dataset-v2/train.csv")
    with force_open("data/dataset-v2/conversation/train.csv", mode="wb") as f:
        df.to_csv(f)
    
    df = message_csv2conversation_csv("data/dataset-v2/test.csv")
    with force_open("data/dataset-v2/conversation/test.csv", mode="wb") as f:
        df.to_csv(f)


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

    register_mappings(ANNModule)
    register_mappings(RnnModule)

    run()
