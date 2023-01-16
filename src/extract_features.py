import datetime as dt
from scipy import sparse
from sentence_transformers import SentenceTransformer
from lib import text_corpus as tc, utils


def extract_features(df, feature_set=(),
                     pretrained=True):  # ['basic', 'linguistic', 'w2v_glove', 'w2v_bert', 'w2v', 'c2v', 'd2v', 'd2v_c']
    tc_df = tc.TextCorpus(df['text'], char_ngram_range=(1, 1), word_ngram_range=(1, 1))
    features = sparse.csr_matrix((0, len(df))).transpose()

    # if 'time' in feature_set:  # Q['time'] ==> unix time stamp
    #     date_format = dt.datetime.strptime(df['time'], "%H:%M")
    #     unix_time = dt.datetime.timestamp(date_format)
    #     features = sparse.csr_matrix(sparse.hstack((
    #         features,
    #         unix_time
    #     )))

    if 'basic' in feature_set:
        features = sparse.csr_matrix(sparse.hstack((
            features,
            tc_df.getLengthsByTerm(),
            tc_df.getCharStat()[0],
            tc_df.getTermStat()[0],
            tc_df.getTfIdF(),
            # tc_df.getCharBM25(),
            # tc_df.getTermBM25()
        )))

    if 'linguistic' in feature_set:
        q_linguistic = sparse.csr_matrix([
            tc_df.getLengths(),
            tc_df.getSpecialCharStat()[0],
            tc_df.hasSpecialChar(),
            tc_df.getUpperCharStat()[0],
            tc_df.getNounStat(),
            tc_df.getVerbStat(),
            tc_df.getAdjectiveStat(),
            tc_df.hasNumber(),
            tc_df.getNumberStat(),
            tc_df.getNonEnglishCharStat()[0],
            tc_df.getAvgLetterPerWord(),
            tc_df.getColemanLiauIndex(),
            tc_df.getDaleChallReadabilityScore(),
            tc_df.getAutomatedReadabilityIndex(),
            tc_df.getDifficultWordsStat(),
            tc_df.getFleschReadingEase(),
            tc_df.getFleschKincaidGrade(),
            tc_df.getGunningFog(),
            tc_df.getLexiconStat(),
            tc_df.getLinsearWriteFormula(),
            tc_df.getSmogIndex(),
            tc_df.getTextStandardLevel()
        ])
        features = sparse.csr_matrix(sparse.hstack((features, q_linguistic.transpose())))

    if 'w2v_glove' in feature_set:
        model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        sentence_embeddings = model.encode(df['text'].values)
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'w2v_bert' in feature_set:
        model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        sentence_embeddings = model.encode(df['text'].values)
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'w2v' in feature_set:
        sentence_embeddings = tc_df.getEmbeddingsByTerm(dim=300, win=1, op='avg')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'c2v' in feature_set:
        sentence_embeddings = tc_df.getEmbeddingsByChar(dim=300, win=1, op='avg')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'd2v' in feature_set:
        sentence_embeddings = tc_df.getEmbeddingsByTerm(dim=300, win=1, op='doc')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    if 'd2v_c' in feature_set:
        sentence_embeddings = tc_df.getEmbeddingsByChar(dim=300, win=1, op='doc')
        features = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

    return features


def extract_load_text_features(df, feature_set, features_file=None):
    try:
        return utils.load_sparse_csr(features_file)
    except FileNotFoundError as e:
        print("File not found! Generating the features matrix ...")
        features = extract_features(df, feature_set)
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
