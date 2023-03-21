# This file is for sake of compatibility between different development environment
from src.niam import main, run
from src.preprocessing import NLTKStopWordRemoving, PunctuationRemoving, RepetitionRemoving
from src.utils.dataset import BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset, CaseSensitiveBertEmbeddingDataset, GloveEmbeddingDataset
from src.models import ANNModule, RnnModule
from settings.mappings import register_mappings, register_mappings_torch



if __name__ == "__main__":
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
