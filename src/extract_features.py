from scipy import sparse
import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer

from lib import text_corpus as tc, utils

def extract_features(Q, feature_set=[], pretrained=True):#['basic', 'linguistic', 'w2v_glove', 'w2v_bert', 'w2v', 'c2v', 'd2v', 'd2v_c']
    """Create sentence embeddings

    Args:
        Q (Dataframe): Training and testing data
        feature_set (list, optional): Word to vector models. Defaults to [].
        pretrained (bool, optional): _description_. Defaults to True.

    Returns:
        CSR Matrix: Sentence embeddings of the training and testing conversations
    """
    tc_Q = tc.TextCorpus(Q['text'], char_ngram_range=(1, 1), word_ngram_range=(1, 1))
    print(len(Q))
    features = sparse.csr_matrix((0,  len(Q))).transpose()
    print('features',features.shape)

    if 'basic' in feature_set: 
        features = sparse.csr_matrix(sparse.hstack((
        features,
        tc_Q.getLengthsByTerm(),
        tc_Q.getCharStat()[0],
        tc_Q.getTermStat()[0],
        tc_Q.getTfIdF(),
        # tc_Q.getCharBM25(),
        # tc_Q.getTermBM25()
        )))

    if 'linguistic' in feature_set:
        q_linguistic = sparse.csr_matrix([
        tc_Q.getLengths(),
        tc_Q.getSpecialCharStat()[0],
        tc_Q.hasSpecialChar(),
        tc_Q.getUpperCharStat()[0],
        tc_Q.getNounStat(),
        tc_Q.getVerbStat(),
        tc_Q.getAdjectiveStat(),
        tc_Q.hasNumber(),
        tc_Q.getNumberStat(),
        tc_Q.getNonEnglishCharStat()[0],
        tc_Q.getAvgLetterPerWord(),
        tc_Q.getColemanLiauIndex(),
        tc_Q.getDaleChallReadabilityScore(),
        tc_Q.getAutomatedReadabilityIndex(),
        tc_Q.getDifficultWordsStat(),
        tc_Q.getFleschReadingEase(),
        tc_Q.getFleschKincaidGrade(),
        tc_Q.getGunningFog(),
        tc_Q.getLexiconStat(),
        tc_Q.getLinsearWriteFormula(),
        tc_Q.getSmogIndex(),
        tc_Q.getTextStandardLevel()
        ])
        features = sparse.csr_matrix(sparse.hstack((features, q_linguistic.transpose())))

    if 'w2v_glove' in feature_set:
        model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        sentence_embeddings = model.encode(Q['text'].values)
        features = sparse.csr_matrix(sparse.hstack((
            features, 
            sentence_embeddings,
            )))

    if 'time' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((features, Q['time'].values.reshape(-1, 1), )))

    if 'count' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((features, Q['msg_word_count'].values.reshape(-1, 1),Q['msg_char_count'].values.reshape(-1,1) )))

    if 'msg_line' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((features, Q['msg_line'].values.reshape(-1, 1), )))
    
    if 'nauthor' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((features, Q['nauthor'].values.reshape(-1, 1), )))
    
    if 'conv_size' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((features, Q['conv_size'].values.reshape(-1, 1), )))

    if 'prv_cat' in feature_set:
        model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        sentence_embeddings = model.encode(Q['prv_cat'].values)
        features = sparse.csr_matrix(sparse.hstack((
            features, 
            sentence_embeddings,
            )))

    if 'nxt_cat' in feature_set:
        model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        sentence_embeddings = model.encode(Q['nxt_cat'].values)
        features = sparse.csr_matrix(sparse.hstack((
            features, 
            sentence_embeddings,
            )))   

    if 'w2v_bert' in feature_set:
        model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        sentence_embeddings = model.encode(Q['text'].values)
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'w2v' in feature_set:
        sentence_embeddings = tc_Q.getEmbeddingsByTerm(dim=300, win=1, op='avg')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'c2v' in feature_set:
        sentence_embeddings = tc_Q.getEmbeddingsByChar(dim=300, win=1, op='avg')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'd2v' in feature_set:
        sentence_embeddings = tc_Q.getEmbeddingsByTerm(dim=300, win=1, op='doc')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'd2v_c' in feature_set:
        sentence_embeddings = tc_Q.getEmbeddingsByChar(dim=300, win=1, op='doc')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    return features

def extract_load_text_features(Q, feature_set, features_file=None):
    """Load saved vectors or create new one

    Args:
        Q (Dataframe): Training and testing data
        feature_set (list[str]): Word to vector models. Defaults to [].
        features_file (str, optional): File to load from. Defaults to None.

    Raises:
        e: Exception or FileNotFoundError

    Returns:
        CSR Matrix: Sentence embeddings of the training and testing conversations
    """
    try:
        return utils.load_sparse_csr(features_file)
    except FileNotFoundError as e:
        print("File not found! Generating the features matrix ...")
        features = extract_features(Q, feature_set)
        print(feature_set)
        utils.save_sparse_csr(features_file, features)
        print(f"saved features with shape (data size, feature size): {features.shape}")
        return features
    except Exception as e:
        raise e

# test
# df = pd.DataFrame([{'text': 'International Organized Crime'}, {'text': 'Hossein Fani'}], index=[0,1])
# features = extract_features(df, feature_set=['basic', 'linguistic', 'w2v_glove', 'w2v_bert'])
# # features = extract_features(df, feature_set=['w2v']) => error
# # features = extract_features(df, feature_set=['c2v'])=> same error
# # features = extract_features(df, feature_set=['d2v'])=> map error
# # features = extract_features(df, feature_set=['d2v_c'])=> map error
#
# print(features)
# features = extract_load_text_features(df, feature_set=['basic', 'linguistic', 'w2v_glove', 'w2v_bert', 'w2v', 'c2v', 'd2v', 'd2v_c'], features_file='test.npz')
# print(features)