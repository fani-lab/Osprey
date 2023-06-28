import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.models import AbstractFeedForward
from settings import settings
from settings.settings import OUTPUT_LAYER_NODES

import logging

logger = logging.getLogger()


class BertBaseUncasedClassifier(AbstractFeedForward):

    def __init__(self, dropout=0.0, *args, **kwargs):
        super(AbstractFeedForward, self).__init__(*args, dimension_list=[], dropout_list=[], **kwargs)
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = dropout
        config.num_labels = 1
        self.core = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        # self.core = AutoModel.from_pretrained("bert-base-uncased") #
        # self.out = torch.nn.Linear(768, OUTPUT_LAYER_NODES, device=self.device)

    def forward(self, x):
        x = self.core(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2])

        return x[0]
    
    @classmethod
    def short_name(cls) -> str:
        return "bert-classifier"
    
    def get_dataloaders(self, dataset, train_ids, validation_ids, batch_size):
        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=train_subsampler, num_workers=3)
        validation_loader = DataLoader(dataset, batch_size=(256 if len(validation_ids) > 1024 else len(validation_ids)),
                                       sampler=validation_subsampler, num_workers=1)
        
        return train_loader, validation_loader
