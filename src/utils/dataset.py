import logging
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import BertTokenizer

from src.preprocessing.base import BasePreprocessing
from src.utils.one_hot_encoder import OneHotEncoder, SequentialOneHotEncoder, SequentialOneHotEncoderWithContext
from src.utils.transformers_encoders import TransformersEmbeddingEncoder, GloveEmbeddingEncoder, SequentialTransformersEmbeddingEncoder, SequentialTransformersEmbeddingEncoderWithContext
from src.utils.commons import nltk_tokenize, force_open, RegisterableObject
from src.preprocessing.author_id_remover import AuthorIDReplacerBert

# from imblearn.over_sampling import SMOTE

logger = logging.getLogger()


class MyDataset(Dataset):
    def __init__(self, train, target):
        if isinstance(train, pd.core.frame.DataFrame):
            self.data = torch.from_numpy(train.values)
            self.labels = torch.from_numpy(target.to_numpy())
        else:
            self.data = torch.stack(train)
            self.labels = torch.from_numpy(target)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    # def oversample_by_smote(self):
    #     smote = SMOTE(random_state=42)
    #     self.data, self.labels = smote.fit_resample(self.data.to_dense(), self.labels)


class BaseDataset(Dataset, RegisterableObject):

    
    def __init__(self, data_path: str, output_path: str, load_from_pkl: bool, apply_record_filter: bool=True,
                 preprocessings: list[BasePreprocessing] = [], persist_data=True, parent_dataset=None, device="cpu", vector_size=-1, *args, **kwargs):
        self.output_path = output_path
        self.parent_dataset = parent_dataset
        self.load_from_pkl = load_from_pkl
        self.preprocessings = preprocessings
        self.persist_data = persist_data
        self.df_path = data_path
        self.device = device
        self.apply_filter = apply_record_filter

        self.__df__ = None
        self.__labels__ = None

        self.already_prepared = False
        
        self.__new_tokens__ = False
        self.__new_encoder__ = False
        self.__new_vectors__ = False

        self.vector_size = vector_size

    @property
    def df(self):
        if self.__df__ is None:
            self.__df__ = pd.read_csv(self.df_path)
            if self.apply_filter:
                self.__df__ = self.filter_records(self.__df__)

        return self.__df__
    
    def vectorize(self, tokens_records, encoder):
        raise NotImplementedError()
    
    def __init_encoder__(self, tokens_records):
        try:
            if self.parent_dataset is not None:
                encoder = self.parent_dataset.encoder
                return encoder
        except Exception as e:
            raise e

        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("encoder.pkl"), "rb") as f:
                encoder = pickle.load(f)
        except FileNotFoundError:
            self.__new_encoder__ = True
            encoder = self.init_encoder(tokens_records)

        return encoder

    def __vectorize__(self, tokens_records, encoder):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("vectors.pkl"), "rb") as f:
                logger.info("loading vectors from file")
                vectors = pickle.load(f)
        except FileNotFoundError:
            logger.info("trying to create vectors from scratch")
            self.__new_vectors__ = True
            vectors = self.vectorize(tokens_records, encoder)
        
        return vectors
    
    def normalize_vector(self, vectors):
        return vectors

    def __str__(self):
        return self.short_name() +"/p" + ".".join([pp.short_name() for pp in self.preprocessings]) + "-v" + str(self.get_vector_size()) +("-filtered" if self.apply_filter else "-nofilter")
    
    def filter_records(self, df):
        logger.info(f"no filter is applied to dataset: {self.short_name()}")
        return df

    def get_session_path(self, filename) -> str:
        return self.output_path + self.__str__() + "/" + filename
    
    def tokenize(self, input) -> list[list[str]]:
        raise NotImplementedError()
    
    def get_labels(self):
        if not self.already_prepared:
            raise ValueError("the dataset is not prepared. Firt run `prepapre` method.")
        
        if self.__labels__ is not None:
            return self.__labels__
        labels = torch.zeros((self.df.shape[0], 1), dtype=torch.float)
        for i in range(len(self.df)):
            labels[i] = self.df.iloc[i]["predatory_conv"]
        self.__labels__ = labels
        return labels

    def get_data(self):
        if not self.already_prepared:
            raise ValueError("the dataset is not prepared. Firt run `prepapre` method.")
        return self.data
    
    def split_dataset_by_label(self, n_splits, split_again, persist_splits=True, stratified=True, load_splits_from=""):
        if load_splits_from is not None and len(load_splits_from) > 0:
            if split_again:
                logger.warning("split again flag is `True`, but it won't be effective because splits are being loaded from a file.")
            logger.info(f"loading splits from: `{load_splits_from}`")
            with open(load_splits_from, "rb") as f:
                splits = pickle.load(f)
                return splits
        splits_path = self.get_session_path(f"splits-n{n_splits}" + ("stratified" if stratified else "") + ".pkl")
        
        try:
            if not split_again:
                with open(splits_path, "rb") as f:
                    splits = pickle.load(f)
                logger.info(f"loading splits from: {splits_path}")
                return splits
        except FileNotFoundError as e:
            logger.warning("could not find the splits file. going to create splits from scratch.")
        
        if stratified:
            labels = self.get_labels()
            data = self.get_data()

            kfolder = StratifiedKFold(n_splits=n_splits, shuffle=True)
        else:
            data = self.get_data()
            labels = None
            kfolder = KFold(n_splits=n_splits, shuffle=True)
        
        splits = [None] * n_splits
        for fold_index, (train_ids, label_ids) in enumerate(kfolder.split(data, labels)):
            splits[fold_index] = (train_ids, label_ids)
        
        if persist_splits:
            with force_open(splits_path, "wb") as f:
                logger.info(f"saving splits at {splits_path}")
                pickle.dump(splits, f)
        logger.info(f"splits created by the following configs: n_splits: `{n_splits}`, stratified: {stratified}, persist_splits: {persist_splits} ")
        return splits

    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("tokens.pkl"), "rb") as f:
                logger.info("trying to load tokens from file")
                tokens = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            self.__new_tokens__ = True
            tokens = self.tokenize(self.df["text"])
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                tokens = [*preprocessor.opt(tokens)]

        return tokens

    def get_vector_size(self, vectors=None):
        if self.vector_size < 0:
            raise ValueError("vector size is not defined or calculated yet")
        return self.vector_size

    def update_vector_size(self, vectors):
        self.vector_size = vectors[0].shape[-1]
        return self.vector_size
    
    def prepare(self):
        if self.already_prepared:
            logger.debug("already called prepared")
            return
        
        if self.parent_dataset is not None:
           self.parent_dataset.prepare()

        tokens = self.preprocess()

        self.encoder = self.__init_encoder__(tokens_records=tokens)

        vectors = self.__vectorize__(tokens, self.encoder)
        vectors = self.normalize_vector(vectors)
        self.update_vector_size(vectors)
        # Persisting changes
        if self.persist_data and self.__new_tokens__:
            tokens_path = self.get_session_path("tokens.pkl")
            logger.info(f"saving tokens as pickle at {tokens_path}")
            with force_open(tokens_path, "wb") as f:
                pickle.dump(tokens, f)
        if self.persist_data and self.__new_vectors__:
            vectors_path = self.get_session_path("vectors.pkl")
            logger.info(f"saving vectors as pickle at {vectors_path}")
            with force_open(vectors_path, "wb") as f:
                pickle.dump(vectors, f)
        if self.persist_data and self.__new_encoder__:
            encoder_path = self.get_session_path("encoder.pkl")
            logger.info(f"saving encoder as pickle at {encoder_path}")
            with force_open(encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)
        
        self.already_prepared = True

        self.labels = self.get_labels()
        
        self.data = vectors
        logger.info("data preparation finished")

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def to(self, device):
        self.labels = self.labels.to(device)
        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(device)

    @property
    def shape(self):
        return (len(self.data), self.get_vector_size())

# It is only for handling fine-tuning
class FineTuningBertDataset(BaseDataset):

    def __init__(self, tokenizer_path="bert-base-uncased", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None

        self.vector_size = 512
    
    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("attention_masks.pkl"), "rb") as f:
                logger.info("trying to load attention_masks from file")
                attention_masks = pickle.load(f)
            with open(self.get_session_path("input_ids.pkl"), "rb") as f:
                logger.info("trying to load input_ids from file")
                input_ids = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            self.__new_tokens__ = True
            tokens = nltk_tokenize(self.df["text"])
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                tokens = [*preprocessor.opt(tokens)]
            input_ids, attention_masks = self.tokenize(tokens)

        return input_ids, attention_masks

    @classmethod
    def short_name(cls) -> str:
        return "finetuning-bert"
    
    def tokenize(self, input) -> tuple[list]:
        logger.info("using BertTokenizer.`bert-base-uncased`")
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=True)

        input_ids = [None] * len(input)
        attention_masks = [None] * len(input)
        for i, tokens in enumerate(input):
            encoded = self.tokenizer.encode_plus(" ".join(tokens), add_special_tokens=True, max_length=512, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
            attention_masks[i] = encoded["attention_mask"]
            input_ids[i]       = encoded["input_ids"]
        input_ids       = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks
    
    def prepare(self):
        if self.already_prepared:
            logger.debug("already called prepared")
            return
        
        
        self.input_ids, self.attention_masks = self.preprocess()

        self.update_vector_size(None)

        if self.persist_data and self.__new_tokens__:
            input_ids_path = self.get_session_path("input_ids.pkl")
            logger.info(f"saving input_ids as pickle at {input_ids_path}")
            with force_open(input_ids_path, "wb") as f:
                pickle.dump(self.input_ids, f)
            attention_mask_path = self.get_session_path("attention_masks.pkl")
            logger.info(f"saving attention_masks as pickle at {attention_mask_path}")
            with force_open(attention_mask_path, "wb") as f:
                pickle.dump(self.attention_masks, f)

        self.already_prepared = True
        self.labels = self.get_labels()
        logger.info("data preparation finished")
    
    def get_vector_size(self, vectors=None):
        return 512 # it is the embedding size of bert-base; check BERT docs

    def update_vector_size(self, vectors):
        self.vector_size = 512
        return self.vector_size

    def __getitem__(self, index):
        return self.attention_masks[index], self.input_ids[index], self.labels[index]

    def to(self, device):
        self.labels = self.labels.to(device)
        
        self.attention_masks = self.attention_masks.to(device)
        self.input_ids = self.input_ids.to(device)

    def __len__(self):
        return len(self.input_ids)
    
    @property
    def shape(self):
        return self.input_ids.shape[0], self.get_vector_size()


class BagOfWordsDataset(BaseDataset):

    @classmethod
    def short_name(cls) -> str:
        return "bow"

    def get_data_generator(self, data, pattern):
        def func():
            for record in data:
                yield pattern(record)

        return func

    def tokenize(self, input) -> list[list[str]]:
        logger.debug("tokenizing using nltk")
        return nltk_tokenize(input)

    def init_encoder(self, tokens_records):
        encoder = OneHotEncoder(vector_size=self.get_vector_size())
        logger.info("started generating bag of words vector encoder")
        data = set()
        data.update(*tokens_records)
        pattern = lambda x: x
        logger.debug("fitting data into one hot encoder")
        encoder.fit(self.get_data_generator(data=data, pattern=pattern))
        return encoder

    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        for record in tokens_records:
            temp = encoder.transform(record=record)
            vectors.append(torch.sparse.sum(torch.cat(temp), dim=0))
        logger.debug("transforming of records into vectors is finished")
        return vectors

    # def oversample_by_smote(self):
    #     smote = SMOTE(random_state=42)
    #     self.data, self.labels = smote.fit_resample(self.data.to_dense(), self.labels)

class ConversationBagOfWords(BagOfWordsDataset):
    
    @classmethod
    def short_name(cls) -> str:
        return "conversation-bow"
    
    def filter_records(self, df):
        logger.info("applying record filtering by 'number_of_authors >= 2' & 'number_of_messages > 6'")
        return df[((df["number_of_authors"] >= 2) & (df["number_of_messages"] > 6))]
    
    def get_labels(self):
        labels = torch.zeros((self.df.shape[0]), dtype=torch.float)
        for i in range(len(self.df)):
            labels[i] = self.df.iloc[i]["predatory_conv"]
        return labels

    def get_data_generator(self, data, pattern):
        def func():
            for record in data:
                for token in record:
                    yield pattern(token)

        return func

    def init_encoder(self, tokens_records):
        encoder = OneHotEncoder(vector_size=self.get_vector_size(), buffer_cap=64)
        logger.info("started generating bag of words vector encoder")
        data = tokens_records
        pattern = lambda x: x
        logger.debug("fitting conversation tokens into one hot encoder")
        encoder.fit(self.get_data_generator(data=data, pattern=pattern))
        return encoder

    def normalize_vector(self, vectors):
        return [vector/torch.sparse.sum(vector) for vector in vectors]


class ConversationBagOfWordsWithTriple(ConversationBagOfWords):
    
    @classmethod
    def short_name(cls) -> str:
        return "conversation-bow-with-triple"
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index


class ConversationBagOfWordsCleaned(ConversationBagOfWords):
    
    def filter_records(self, df):
        logger.info("applying record filtering by 'number_of_authors == 2' & 'number_of_messages > 3'")
        return df[((df["number_of_authors"] == 2) & (df["number_of_messages"] > 3))]

    @classmethod
    def short_name(cls) -> str:
        return "conversation-bow-cleaned"


class CNNConversationBagOfWords(ConversationBagOfWords):

    @classmethod
    def short_name(cls) -> str:
        return "cnn-conversation-bow"
    
    def init_encoder(self, tokens_records):
        logger.info("started generating bag of words vector encoder")
        encoder = OneHotEncoder(vector_size=self.get_vector_size(), buffer_cap=64, vectors_dimensions=3)
        data = tokens_records
        pattern = lambda x: x
        logger.debug("fitting conversation tokens into one hot encoder")
        encoder.fit(self.get_data_generator(data=data, pattern=pattern))
        return encoder

    @property
    def shape(self):
        return (len(self.data), self.data[0].shape[0])


class TimeBasedBagOfWordsDataset(BagOfWordsDataset):
    
    @classmethod
    def short_name(cls) -> str:
        return "time-bow"

    def get_normalization_params(self, columns):
        if self.parent_dataset is not None:
            return self.parent_dataset.get_normalization_params(columns)
        if hasattr(self, "normalization_params"):
            return self.normalization_params

        self.normalization_params = dict()
        for c in columns:
            self.normalization_params[c] = (self.df[c].mean(), self.df[c].std())

        return self.normalization_params

    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        context_columns = ("nauthor", "msg_line", "time")
        self.df[["nauthor", "msg_line", "time"]]
        normalization_params = self.get_normalization_params(context_columns)
        for c in context_columns:
            self.df[f"normalized_{c}"] = (self.df[c] - normalization_params[c][0]) / normalization_params[c][1]
        context_indices = [list(range(len(context_columns)))]

        for i, record in enumerate(tokens_records):
            onehots = torch.sparse.sum(torch.cat(encoder.transform(record=record)), dim=0)
            context = torch.sparse_coo_tensor(context_indices,
                                              [self.df.iloc[i][f"normalized_{c}"] for c in context_columns],
                                              (len(context_columns),), dtype=torch.float32)

            vectors.append(torch.hstack((context, onehots)))
        logger.debug("transforming of records into vectors is finished")
        return vectors


class UncasedBaseBertTokenizedDataset(BaseDataset, RegisterableObject):

    @classmethod
    def short_name(cls) -> str:
        return "bert-based-uncased"
    
    def init_encoder(self, tokens_records):
        encoder = BertTokenizer.from_pretrained('bert-base-uncased')
        logger.debug("uncased bert-base tokenizer is being used as encoder")
        return encoder

    def tokenize(self, input):
        return nltk_tokenize(input)

    def vectorize(self, tokens_records, encoder):
        vectors = [None] * len(tokens_records)
        for i, record in enumerate(tokens_records):
            temp = encoder(" ".join(record), return_tensors="pt", padding='max_length', truncation=True)
            vectors[i] = {k: temp[k].squeeze() for k in temp.keys()}
        self.__new_encoder__ = False
        return vectors

    def to(self, device):
        self.labels = self.labels.to(device)
        keys = self.data[0].keys()
        for i in range(len(self.data)):
            for k in keys:
                self.data[i][k].to(device)
    
    def get_vector_size(self, vectors=None):
        return 768 # it is the embedding size of bert-base; check BERT docs

    def update_vector_size(self, vectors):
        self.vector_size = 768
        return self.vector_size

    def __getitem__(self, index):
        return (self.data[index]["input_ids"], self.data[index]["attention_mask"], self.data[index]["token_type_ids"]), self.labels[index]

class TransformersEmbeddingDataset(BaseDataset, RegisterableObject):

    @classmethod
    def short_name(cls) -> str:
        return "conversation-distilroberta-v1"
        
    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = TransformersEmbeddingEncoder(transformer_identifier="sentence-transformers/all-distilroberta-v1", device=self.device)
        return encoder

    def tokenize(self, input):
        logger.debug("tokenizing using nltk")
        return nltk_tokenize(input)

    def vectorize(self, tokens_records, encoder):
        vectors = [None] * len(tokens_records)
        for i, record in enumerate(tokens_records):
            vectors[i] = torch.cat(encoder.transform(record))
        return vectors

    def get_vector_size(self, vectors=None):
        return 768

    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["number_of_authors"] >= 2) & (df["number_of_messages"] > 6)]


class UncasedBaseBertEmbeddingDataset(TransformersEmbeddingDataset):
    
    @classmethod
    def short_name(cls) -> str:
        return "conversation-bert-base-uncased"
        
    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = TransformersEmbeddingEncoder(transformer_identifier="bert-base-uncased", device=self.device)
        return encoder

# Do not use it for now
class FineTunedBertEmbeddingDataset(TransformersEmbeddingDataset):

    def __init__(self, transformer_identifier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("this module is not implemented completely.")
        # `transformer_identifier` should be a path 
        # ex: "data/embeddings/finetuned/finetuning-bert/psw.rr.idr-v512-nofilter/"
        self.transformer_identifier = transformer_identifier

    @classmethod
    def short_name(cls) -> str:
        return "embedding/user-pretrained"
        
    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = TransformersEmbeddingEncoder(transformer_identifier=self.transformer_identifier, device=self.device)
        return encoder


class CaseSensitiveBertEmbeddingDataset(TransformersEmbeddingDataset):
    
    @classmethod
    def short_name(cls) -> str:
        return "embedding/bert-base-cased"
    
    def init_encoder(self, tokens_records):
        encoder = TransformersEmbeddingEncoder(transformer_identifier="bert-base-cased", device=self.device)

        return encoder

    def get_session_path(self, filename) -> str:
        return self.output_path + "tranformer/bert-base-cased/" + filename


class GloveEmbeddingDataset(BaseDataset, RegisterableObject):
    
    @classmethod
    def short_name(cls) -> str:
        return "glove/twitter.50d"
        
    def init_encoder(self, tokens_records):
        logger.debug("Glove Embedding Dataset being initialized")
        path = "data/embeddings/glove.twitter.27B/glove.twitter.27B.50d.txt"
        encoder = GloveEmbeddingEncoder(path)
        return encoder

    def tokenize(self, input):
        logger.debug("tokenizing using nltk")
        return nltk_tokenize(input)

    def vectorize(self, tokens_records, encoder):
        vectors = [None] * len(tokens_records)
        for i, record in enumerate(tokens_records):
            vectors[i] = torch.cat(encoder.transform(record))
        return vectors


class SequentialConversationDataset(BaseDataset): # TODO: checkout to(device) method
    """
    a dataset where each record is a sorted sequence of any size and each record has one label
    """
    def __init__(self, data_path: str, output_path: str, load_from_pkl: bool, apply_record_filter: bool = True, preprocessings: list[BasePreprocessing] = [], persist_data=True, parent_dataset=None, device="cpu", *args, **kwargs):
        super().__init__(data_path, output_path, load_from_pkl, apply_record_filter, preprocessings, persist_data, parent_dataset, device, *args, **kwargs)
        self.__sequence__ = None

    # def filter_records(self, df):
    #     logger.info("applying record filtering by 'nauthor == 2'")
    #     return df[(df["nauthor"] == 2)]

    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["nauthor"] >= 2) & (df["conv_size"] > 6)]

    @property
    def sequence(self):
        if self.__sequence__ is None:
            df = self.df
            self.__sequence__ = df.sort_values("msg_line").groupby("conv_id")
        return self.__sequence__

    def get_data_generator(self, data, pattern):
        def func():
            for sequence in data:
                for record in sequence:
                    for token in record:
                        yield pattern(token)

        return func

    @classmethod
    def short_name(cls) -> str:
        return "basic-sequential"

    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("tokens.pkl"), "rb") as f:
                logger.info("trying to load tokens from file")
                messages = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            self.__new_tokens__ = True
            messages = [self.tokenize(g["text"]) for k, g in self.sequence]
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                messages = [[*preprocessor.opt(sequence)] for sequence in messages]
        return messages
    
    def tokenize(self, input) -> list[list[str]]:
        return nltk_tokenize(input)

    def init_encoder(self, tokens_records):
        encoder = SequentialOneHotEncoder(vector_size=self.get_vector_size())
        logger.info("started generating bag of words vector encoder")
        pattern = lambda x: x
        logger.debug("fitting data into one hot encoder")
        encoder.fit(self.get_data_generator(data=tokens_records, pattern=pattern))
        return encoder

    def vectorize(self, tokens_records: list[list[str]], encoder):
        logger.info("vectorizing message records")
        vectors = []
        for record in tokens_records:
            sequence = encoder.transform(record=record)
            temp = torch.stack([torch.sparse.sum(torch.cat(t), dim=0) for t in sequence])
            vectors.append(temp)
        logger.debug("vectorizing finished")
        
        return vectors
    
    def get_labels(self):
        if not self.already_prepared:
            raise ValueError("the dataset is not prepared. Firt run `prepapre` method.")
        if self.__labels__ is not None:
            return self.__labels__
        labels = torch.zeros((len(self.sequence), 1), dtype=torch.float)

        for i, (_, group) in enumerate(self.sequence):
            labels[i] = group.iloc[0]["predatory_conv"]
        self.__labels__ = labels
        return labels

    @property
    def shape(self):
        return (len(self.data), -1, self.data[0].shape[-1]) # Fix the last shape with get_vector_size


class BaseContextualSequentialConversationOneHotDataset(SequentialConversationDataset):
    CONTEXT_LENGTH = 0
    @classmethod
    def short_name(cls) -> str:
        return "contextual-onehot-sequential"
    
    def get_data_generator(self, data, pattern):
        def func():
            for _, sequence in data:
                for record in sequence:
                    for token in record:
                        yield pattern(token)

        return func
    
    def init_encoder(self, tokens_records):
        encoder = SequentialOneHotEncoderWithContext(context_length=self.CONTEXT_LENGTH, vector_size=self.get_vector_size(), )
        logger.info("started generating sequential-conversation bag of words vector encoder")
        pattern = lambda x: x
        logger.debug("fitting data into one hot encoder")
        encoder.fit(self.get_data_generator(data=tokens_records, pattern=pattern))
        return encoder

    def tokenize(self, sequence):
        raise NotImplementedError()

    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("tokens.pkl"), "rb") as f:
                logger.info("trying to load tokens from file")
                messages = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            self.__new_tokens__ = True
            messages = self.tokenize(self.sequence)
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                messages = [(context, tuple(preprocessor.opt(sequence))) for context, sequence in messages]
        return messages
    
    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        
        for i, record in enumerate(tokens_records):
            sequence = encoder.transform(record=record)
            onehots = torch.stack([torch.sparse.sum(torch.cat(t), dim=0) for t in sequence])

            vectors.append(onehots)

        logger.debug("transforming of records into vectors is finished")
        return vectors


class TemporalSequentialConversationOneHotDataset(BaseContextualSequentialConversationOneHotDataset):
    
    CONTEXT_LENGTH = 1
    
    @classmethod
    def short_name(cls) -> str:
        return "temporal-sequential"
    
    def tokenize(self, sequence):
        messages = [None] * len(sequence)
        for i, (k, g) in enumerate(sequence):
            temp = np.floor(g["time"].tolist())
            messages[i] = (((temp*60 + (g["time"].tolist()- temp)*100)/1440,), nltk_tokenize(g["text"]))
        return messages

class TemporalAuthorsSequentialConversationOneHotDataset(BaseContextualSequentialConversationOneHotDataset):
    
    CONTEXT_LENGTH = 2
    
    @classmethod
    def short_name(cls) -> str:
        return "time-nauthor-sequential"

    def tokenize(self, sequence):
        messages = [None] * len(sequence)
        for i, (k, g) in enumerate(sequence):
            temp = np.floor(g["time"])
            messages[i] = ((((temp*60 + (g["time"]- temp)*100)/1440).tolist(), (g["nauthor"]/4.0).tolist(),), nltk_tokenize(g["text"]))
        return messages


class TemporalSequentialConversationOneHotDatasetFiltered(TemporalSequentialConversationOneHotDataset):
    
    @classmethod
    def short_name(cls) -> str:
        return "time-sequential-bow-convsize"

    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["nauthor"] >= 2) & (df["conv_size"] > 6)]


class TemporalAuthorsSequentialConversationOneHotDatasetFiltered(TemporalAuthorsSequentialConversationOneHotDataset):

    @classmethod
    def short_name(cls) -> str:
        return "time-nauthor-sequential-bow-convsize"

    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["nauthor"] >= 2) & (df["conv_size"] > 6)]


class SequentialConversationDatasetFiltered(SequentialConversationDataset):

    @classmethod
    def short_name(cls) -> str:
        return "sequential-bow-convsize"

    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["nauthor"] >= 2) & (df["conv_size"] > 6)]


class SequentialConversationEmbeddingDataset(SequentialConversationDataset):
    
    def filter_records(self, df):
        logger.info("applying record filtering by 'nauthor >= 2 & conv_size > 6'")
        return df[(df["nauthor"] >= 2) & (df["conv_size"] > 6)]

    @classmethod
    def short_name(cls) -> str:
        return "sequential-embedding"

    def vectorize(self, tokens_records: list[list[str]], encoder):
        logger.info("vectorizing message records")
        vectors = []
        for record in tokens_records:
            sequence = encoder.transform(record=record)
            temp = torch.stack([torch.cat(t) for t in sequence])
            vectors.append(temp)
        logger.debug("vectorizing finished")
        
        return vectors
    
    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = SequentialTransformersEmbeddingEncoder(transformer_identifier="all-distilroberta-v1", device=self.device)
        return encoder

    def get_vector_size(self, vectors=None):
        return 768


class SequentialConversationUniversalSentenceEncoderDataset(SequentialConversationEmbeddingDataset):
    # https://huggingface.co/sentence-transformers/use-cmlm-multilingual
    
    @classmethod
    def short_name(cls) -> str:
        return "use-embedding"

    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = SequentialTransformersEmbeddingEncoder(transformer_identifier="sentence-transformers/use-cmlm-multilingual", device=self.device)
        return encoder

    def get_vector_size(self, vectors=None):
        return 768 # it is the same as some other baselines. visit the huggingface of the transformer for more info


class BaseContextualSequentialConversationEmbeddingDataset(SequentialConversationEmbeddingDataset):
    CONTEXT_LENGTH = 0

    @classmethod
    def short_name(cls) -> str:
        return "contextual-ebedding-sequential"
    
    def init_encoder(self, tokens_records):
        logger.debug("initializing sequential transformer embedding encoder with context: all-distilroberta-v1")
        encoder = SequentialTransformersEmbeddingEncoderWithContext(context_length=self.CONTEXT_LENGTH, transformer_identifier="all-distilroberta-v1", device=self.device)
        return encoder
    
    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("tokens.pkl"), "rb") as f:
                logger.info("trying to load tokens from file")
                messages = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            self.__new_tokens__ = True
            messages = self.tokenize(self.sequence)
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                messages = [[context, tuple(preprocessor.opt(sequence))] for context, sequence in messages]
        return messages

    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        
        for i, record in enumerate(tokens_records):
            sequence = encoder.transform(record=record)
            embeddings = torch.stack(sequence).float()
            vectors.append(embeddings)

        logger.debug("transforming of records into vectors is finished")
        return vectors


class TemporalSequentialConversationEmbeddingDataset(BaseContextualSequentialConversationEmbeddingDataset):

    CONTEXT_LENGTH = 1

    @classmethod
    def short_name(cls) -> str:
        return "temporal-sequential-embedding"
    
    def tokenize(self, sequence):
        messages = [None] * len(sequence)
        for i, (k, g) in enumerate(sequence):
            temp = np.floor(g["time"].tolist())
            messages[i] = (((temp*60 + (g["time"].tolist()- temp)*100)/1440,), nltk_tokenize(g["text"]))
        return messages


class TemporalAuthorsSequentialConversationEmbeddingDataset(BaseContextualSequentialConversationEmbeddingDataset):
    
    CONTEXT_LENGTH = 2
    
    @classmethod
    def short_name(cls) -> str:
        return "temporal-nauthor-sequential-embedding"

    def tokenize(self, sequence):
        messages = [None] * len(sequence)
        for i, (k, g) in enumerate(sequence):
            temp = np.floor(g["time"])
            messages[i] = ((((temp*60 + (g["time"] - temp)*100)/1440).tolist(), (g["nauthor"]/4.0).tolist(),), nltk_tokenize(g["text"]))
        return messages


class SequentialConversationBertBaseDataset(SequentialConversationEmbeddingDataset):
    @classmethod
    def short_name(cls) -> str:
        return "sequential-bert-base"
    
    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = SequentialTransformersEmbeddingEncoder(transformer_identifier="bert-base-uncased", device=self.device)
        return encoder
    
    def get_vector_size(self, vectors=None):
        return 768


class TemporalSequentialConversationBertBaseDataset(TemporalSequentialConversationEmbeddingDataset):

    @classmethod
    def short_name(cls) -> str:
        return "temporal-sequential-bert-base"

    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = SequentialTransformersEmbeddingEncoderWithContext(context_length=self.CONTEXT_LENGTH, transformer_identifier="bert-base-uncased", device=self.device)
        return encoder

    def get_vector_size(self, vectors=None):
        return 768


class TemporalAuthorsSequentialConversationBertBaseDataset(TemporalAuthorsSequentialConversationEmbeddingDataset):

    @classmethod
    def short_name(cls) -> str:
        return "temporal-nauthor-sequential-bert-base"

    def init_encoder(self, tokens_records):
        logger.debug("Transformer Embedding Dataset being initialized")
        encoder = SequentialTransformersEmbeddingEncoderWithContext(context_length=self.CONTEXT_LENGTH, transformer_identifier="bert-base-uncased", device=self.device)
        return encoder

    def get_vector_size(self, vectors=None):
        return 768