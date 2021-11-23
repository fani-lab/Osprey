# coding: utf-8
import re
import string
import matplotlib as mpl
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import numpy as np
import itertools
import multiprocessing
import sys
sys.path.extend(["../"])

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec, Word2Vec
from lib.textstat.textstat import textstat as ts
from lib.bm25.bm25 import BM25Transformer
from lib.unicode_blocks import UnicodeBlock
from lib.part_of_speech import Pos

np.random.seed(7881)

class TextCorpus:
    def __init__(self, corpus, char_ngram_range=(1,1), word_ngram_range=(1,1), words=None, chars=None):
        self.corpus = corpus
        self.word_pattern = r"\b[a-zA-Z0-9\.]*[a-zA-Z]+[0-9\.]*\b"
        cv = CountVectorizer(stop_words='english', analyzer='word', lowercase=True, vocabulary=words,
                             token_pattern=self.word_pattern, ngram_range=word_ngram_range)
        self._word_ngram_range = cv.ngram_range
        self._word_count = cv.fit_transform(self.corpus)
        self.words = cv.vocabulary_
        self.inv_words = {v: k for k, v in self.words.items()}
        self._word_freq = TfidfTransformer(use_idf=False).fit_transform(self._word_count)

        if char_ngram_range is not None:
            cv = CountVectorizer(lowercase=False, vocabulary=chars, ngram_range=char_ngram_range, analyzer='char_wb')
            self._char_ngram_range = cv.ngram_range
            self._char_count = cv.fit_transform(self.corpus)
            self.chars = cv.vocabulary_
            self.inv_chars = {v: k for k, v in self.chars.items()}
            self._char_freq = TfidfTransformer(use_idf=False).fit_transform(self._char_count)
        self._pos = None

    def getLengths(self):
        """length of docs by unicode char"""
        return [len(doc) for doc in self.corpus]

    def getLengthsByTerm(self):
        """length of docs by terms"""
        if self._word_ngram_range == (1,1):
            return [row.count_nonzero() for row in self._word_count]
        else:
            cv = CountVectorizer(stop_words='english', analyzer='word', lowercase=True, token_pattern=self.word_pattern)
            return [row.count_nonzero() for row in cv.fit_transform(self.corpus)]

    def getCharStat(self):
        """[count, freq]"""
        return self._char_count, self._char_freq

    def getSpecialCharStat(self):
        """special character [count, freq]"""
        count = []
        freq = []
        specialCharIdx = [self.chars[c] for c in string.punctuation if c in self.chars]
        count = self._char_count[:, specialCharIdx].sum(axis=1).flatten().tolist()[0]
        freq = self._char_freq[:, specialCharIdx].sum(axis=1).flatten().tolist()[0]
        return count, freq

    def hasSpecialChar(self):
        return [True if c > 0 else False for c in self.getSpecialCharStat()[0]]

    def getUpperCharStat(self):
        """Upper character [count, freq]"""
        count = []
        freq = []
        upperCharIdx = [self.chars[c] for c in map(chr, range(65, 91)) if c in self.chars]
        count = self._char_count[:, upperCharIdx].sum(axis=1).flatten().tolist()[0]
        freq = self._char_freq[:, upperCharIdx].sum(axis=1).flatten().tolist()[0]
        return count, freq

    def getCfIdf(self):
        """character-based tfidf"""
        pass

    def getTermStat(self):
        """[count, freq]"""
        return self._word_count, self._word_freq

    def getTfIdF(self):
        """tfidf matrix"""
        return TfidfVectorizer(stop_words='english',
                               analyzer='word',
                               vocabulary=self.words,
                               token_pattern=self.word_pattern).fit_transform(
            self.corpus)

    def getPosStat(self):
        """part of speech count by type"""
        if self._pos is None:
            tokens = [nltk.word_tokenize(doc) for doc in self.corpus]
            tags = [nltk.pos_tag(i) for i in tokens]
            c = []
            for i in tags:
                c.append(Counter([v if v not in string.punctuation else "PUNCT" for k, v in i]))
            # for each doc we have pos count e.g.:
            #  c = [Counter({'DT': 2, 'JJ': 1, 'NN': 2, 'PUNCT': 1, 'VBZ': 1}),
            #       Counter({'DT': 2, 'JJ': 1, 'NN': 0, 'PUNCT': 5, 'VBZ': 3})]
            # let's reverse the dic for all POS i.e. r[i][j] return the count of pos i in doc_j. e.g.:
            #  r = {'CC': [0, 0], 'CD': [0, 0], 'DT': [2, 2],'NN': [2, 2],'PRP': [0, 0],'PUNCT': [1, 1],'VBZ': [1, 1], ...}
            r = {}
            for i in Pos:
                r[i] = [j[i.value] for j in c]
            self._pos = r;
        return self._pos;

    def getNounStat(self):
        r = self.getPosStat();
        return list(map(sum, zip(r[Pos.NounPlural],
                            r[Pos.NounSingularOrMass],
                            r[Pos.ProperNounPlural],
                            r[Pos.ProperNounSingular])))

    def getVerbStat(self):
        r = self.getPosStat();
        return list(map(sum, zip(r[Pos.Verb3rdPersonSingularPresent],
                            r[Pos.VerbBaseForm],
                            r[Pos.VerbGerundOrPresentParticiple],
                            r[Pos.VerbNon3rdPersonSingularPresent],
                            r[Pos.VerbPastParticiple],
                            r[Pos.VerbPastTense])))

    def getAdjectiveStat(self):
        r = self.getPosStat();
        return list(map(sum,
                   zip(r[Pos.Adjective], r[Pos.AdjectiveComparative], r[Pos.AdjectiveSuperlative])))

    def getNumberStat(self):
        """number [count, ?]"""
        return [len(re.findall(r"[0-9\.]+", doc)) for doc in self.corpus]

    def hasNumber(self):
        return [True if c > 0 else False for c in self.getNumberStat()]

    def getCharStatByLanguage(self):
        r = {}
        for block in UnicodeBlock:
            charIdx = [self.chars[c] for c in self.chars.keys() if re.findall(block.value, c)]
            count = self._char_count[:, charIdx].sum(axis=1).flatten().tolist()[0]
            freq = self._char_freq[:, charIdx].sum(axis=1).flatten().tolist()[0]
            r[block] = count, freq
        return r

    def hasTamilChar(self):
        return [True if c > 0 else False for c in self.getCharStatByLanguage()[UnicodeBlock.Tamil][0]]

    def hasChineseChar(self):
        return [True if c > 0 else False for c in self.getCharStatByLanguage()[UnicodeBlock.CJKUnifiedIdeographs][0]]

    def getNonEnglishCharStat(self):
        charIdx = [self.chars[c] for c in self.chars.keys() if not re.findall(UnicodeBlock.BasicLatin.value, c)]
        count = self._char_count[:, charIdx].sum(axis=1).flatten().tolist()[0]
        freq = self._char_freq[:, charIdx].sum(axis=1).flatten().tolist()[0]
        return count, freq

    def getTermStatByLanguage(self):
        """non english [count, freq]."""
        # for now, the self.words only has english tokens!
        # r = {}
        # for block in UnicodeBlock:
        #     termIdx = [self.words[c] for c in self.words.keys() if re.findall(block.value, c)]
        #     count = self._word_count[:, termIdx].sum(axis=1)
        #     freq = self._word_freq[:, termIdx].sum(axis=1)
        #     r[block] = count, freq
        # return r
        pass

    def getNonEnglishTermStat(self):
        # for now, the self.words only has english tokens!
        # termIdx = [self.words[c] for c in self.words.keys() if not re.findall(UnicodeBlock.BasicLatin.value, c)]
        # count = self._word_count[:, termIdx].sum(axis=1)
        # freq = self._word_freq[:, termIdx].sum(axis=1)
        # return count, freq
        pass

    def getRareTerms(self, min_f=3):
        rare_word_idx = np.where(self._word_count.sum(axis=0) < min_f)
        return [self.inv_words[i] for i in rare_word_idx[1]]

    def getColorStat(self):
        colorIdx = [self.words[c] for c in mpl.colors.cnames.keys() if self.words.has_key(c)]
        count = self._word_count[:, colorIdx].sum(axis=1).flatten().tolist()[0]
        freq = self._word_freq[:, colorIdx].sum(axis=1).flatten().tolist()[0]
        return count, freq

    def getBrandStat(self, brands):
        # brandIdx = [self.words[c] for c in self.words.keys() if c in brands]
        # return self._word_count[:, brandIdx].sum(axis=1).flatten().tolist()[0]#, [c for c in self.words.keys() if c in map(lambda x: x.lower(), brands)]

        r = []
        for doc in self.corpus:
            r.append(sum([1 for c in brands if doc.lower().find(c) > -1]))
        return r

    def getEmbeddingsByTerm(self, dim=50, win=1, pretrained_file=None, binary=False, word_cleaning=False, op='doc'): #op={doc, avg, sum}
        d2v = []
        if op == 'doc':#doc2vec
            d2v_model = TextCorpus._getDoc2Vec(self._word_count.toarray(), dim, win)
            # w2v = []
            # for i in xrange(len(d2v_model.vocab)):
            #     w2v.append(d2v_model[str(1)])
            for i in range(len(d2v_model.docvecs.doctags)):
                d2v.append(d2v_model.docvecs[str(i)])
        else: #word2vec and then apply op on words of each doc
            w2v = None
            if pretrained_file is not None:
                w2v = Word2Vec.load_word2vec_format(pretrained_file, binary=binary)
            else:
                w2v = Word2Vec(self._word_count.toarray(), vector_size=dim, window=win, min_count=0, workers=multiprocessing.cpu_count())

            if op == 'sum':
                func = np.sum
            elif op == 'avg':
                func = np.average
            if word_cleaning:#shouldn't be used when there is no pretrained!
                np.apply_along_axis(lambda x: d2v.append(
                    func([w2v[re.sub("[^a-zA-Z]", " ", self.inv_words[i]).strip()] for i in x.nonzero()[0]
                          if w2v.__contains__(re.sub("[^a-zA-Z]", " ", self.inv_words[i]).strip())], axis=0) if len(x.nonzero()[0]) > 0 else np.zeros(dim)),
                                    arr=self._word_count.toarray(), axis=1)
            else:
                np.apply_along_axis(lambda x: d2v.append(
                    func([w2v[str(i)] for i in x.nonzero()[0] if w2v.__contains__(str(i))], axis=0) if len(x.nonzero()[0]) > 0 else np.zeros(dim)), arr=self._word_count.toarray(), axis=1)
        return np.array(d2v)

    def getEmbeddingsByChar(self, dim=50,win=1, pre=None, op='doc'): #avg/sum
        d2v = []
        if op == 'doc':#doc2vec
            d2v_model = TextCorpus._getDoc2Vec(self._char_count.toarray(), dim, win)
            for i in range(len(d2v_model.docvecs.doctags)):
                d2v.append(d2v_model.docvecs[str(i)])
            return np.array(d2v) #, np.array(w2v)
        else:
            w2v = Word2Vec(self._char_count.toarray(), vector_size=dim, window=win,
                               min_count=0, workers=multiprocessing.cpu_count())
            if op == 'sum':
                func = np.sum
            elif op == 'avg':
                func = np.average
            np.apply_along_axis(lambda x: d2v.append(
                    func([w2v[str(i)] for i in x.nonzero()[0] if w2v.__contains__(str(i))], axis=0) if len(x.nonzero()[0]) > 0 else np.zeros(dim)),
                                    arr=self._char_count.toarray(), axis=1)
        return np.array(d2v)

    @staticmethod
    def _getDocsByBagOfTokenIds(docs_tokens):
        docs = []
        np.apply_along_axis(lambda x: docs.append(
            map(str, list(itertools.chain.from_iterable(itertools.repeat(i, x[i]) for i in x.nonzero()[0])))),
                            arr=docs_tokens, axis=1)
        return docs

    @staticmethod
    def _getDoc2Vec(docs_tokens, dim, win):
        docs = TextCorpus._getDocsByBagOfTokenIds(docs_tokens)
        tds = []
        for i, doc in enumerate(docs):
            tds.append(TaggedDocument(doc, [str(i)]))

        # alpha=0.025, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1,
        # negative=0, dbow_words=0, dm_mean=0, dm_concat=0, dm_tag_count=1, docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None
        model = Doc2Vec(tds, dm=1, vector_size=dim, window=win, min_alpha=0.025, min_count=0, workers=multiprocessing.cpu_count())
        # start training
        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s' % epoch)
            model.train(tds)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        return model

    def getCharBM25(self, k1=2.0, b=0.75):
        return BM25Transformer(k1=k1, b=b).fit(self._char_count).transform(self._char_count)

    def getTermBM25(self, k1=2.0, b=0.75):
        return BM25Transformer(k1=k1, b=b).fit(self._word_count).transform(self._word_count)

    #Readability features
    def getSyllableStat(self):
        return [ts.syllable_count(doc) for doc in self.corpus]

    def getPolysyllabStat(self):
        return [ts.polysyllabcount(doc) for doc in self.corpus]

    def getAvgLetterPerWord(self):
        return [ts.avg_letter_per_word(doc) for doc in self.corpus]

    def getAvgSentencePerWord(self):
        return [ts.avg_sentence_per_word(doc) for doc in self.corpus]

    def getAvgSyllablePerWord(self):
        return [ts.avg_syllables_per_word(doc) for doc in self.corpus]

    def getAutomatedReadabilityIndex(self):
        return [ts.automated_readability_index(doc) for doc in self.corpus]

    def getColemanLiauIndex(self):
        return [ts.coleman_liau_index(doc) for doc in self.corpus]

    def getDaleChallReadabilityScore(self):
        return [ts.dale_chall_readability_score(doc) for doc in self.corpus]

    def getDifficultWordsStat(self):
        return [ts.difficult_words(doc) for doc in self.corpus]

    def getFleschReadingEase(self):
        return [ts.flesch_reading_ease(doc) for doc in self.corpus]

    def getFleschKincaidGrade(self):
        return [ts.flesch_kincaid_grade(doc) for doc in self.corpus]

    def getGunningFog(self):
        return [ts.gunning_fog(doc) for doc in self.corpus]

    def getLexiconStat(self):
        return [ts.lexicon_count(doc) for doc in self.corpus]

    def getLinsearWriteFormula(self):
        return [ts.linsear_write_formula(doc) for doc in self.corpus]

    def getSmogIndex(self):
        return [ts.smog_index(doc) for doc in self.corpus]

    def getTextStandardLevel(self):
        return [ts.text_standard(doc) for doc in self.corpus]

    def permute(self, set):
        setIdx = [self.words[c] for c in set if self.words.has_key(c)]
        result = []
        for rowidx, row in enumerate(self._word_count):
            inset = [e for e in set if self.words.has_key(e) and self.words[e] in [i for i in setIdx if row[0, i]] > 0]
            outset = [e for e in set if e not in inset]
            newtext = []
            for e in inset:
                patt = re.compile(e, re.IGNORECASE)
                for ee in outset:
                    newtext.append(patt.sub(ee, self.corpus[rowidx]))
            result.append(newtext)
        return result


if __name__ == '__main__':

    # unit tests
    tc = TextCorpus([u'ANMYNA ANMYNA ANMYNA Complaint blue Silky blue Set 柔顺洗发配套 (Shampoo  520ml + Conditioner 250ml)',
                     # u'ANMYNA ANMYNA ANMYNA Complaint blue Silky Set 柔顺洗发配套 (Shampoo 520ml + Conditioner 250ml)',
                     u'YBC-Mini Slim 2.4G USB Wireless Optical Mouse for Computer PC (Red)',
                     u'Vanker New Multifunctional Practical Magic blue Headband Mask Turban Scarf Hood Chic Bike Bicycle Cycle Outdoors Sport HOT'], char_ngram_range=(1,5), word_ngram_range=(1,1))

    print(tc.getLengths())
    print(tc.getLengthsByTerm())
    # print tc.getRareTerms()

    # d2v_t = tc.getEmbeddingsByTerm(dim=50, win=1, op='sum')
    # print(d2v_t)
    # d2v_c = tc.getEmbeddingsByTerm(dim=50, win=1, op='avg')
    # print(d2v_c)
    # d2v_t = tc.getEmbeddingsByChar(dim=50, win=1, op='sum')
    # print(d2v_t)
    # d2v_c = tc.getEmbeddingsByChar(dim=50, win=1, op='avg')
    # print(d2v_c)
    # d2v = tc.getEmbeddingsByTerm(pretrained_file='../../Dataset/pretrained/glove.6B.50d.txt.gensim', binary=False, word_cleaning=False, op='sum')
    # print d2v
    # d2v = tc.getEmbeddingsByTerm(pretrained_file='../../Dataset/pretrained/GoogleNews-vectors-negative300.bin', binary=True, word_cleaning=True, op='sum')
    # print d2v
    #
    # d2v_t = tc.getEmbeddingsByTerm()
    # print d2v
    #
    # d2v_c = tc.getEmbeddingsByChar()
    # print d2v
    #
    # from scipy import sparse
    # features = sparse.csr_matrix(sparse.hstack((
    #     sparse.csr_matrix(d2v_c),d2v_t)))
    # #
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=5)
    # features = sparse.csr_matrix(sparse.hstack((
    #     sparse.csr_matrix(pca.fit_transform(tc.getCharStat()[0].toarray())), pca.fit_transform(tc.getTermStat()[0].toarray()))))
    #
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=5)
    # features = sparse.csr_matrix(sparse.hstack((
    # sparse.csr_matrix(tsne.fit_transform(tc.getCharStat()[0].toarray())), tsne.fit_transform(tc.getTermStat()[0].toarray()))))


    # print tc.getCharBM25()
    # print tc.getTermBM25()

    # from scipy import sparse
    # sparse.csr_matrix(sparse.hstack((sparse.csr_matrix(tc.getCharLSA()), tc.getTermLSA())))

    # set = ['red','blue', 'green', 'yellow', 'black']
    # re = tc.permute(set)
    # print re

    # maxidx = tc.getTermStat()[0].argmax(axis=1)
    # tc_valid = TextCorpus([
    #                           u'[International Shipping]Giazzuro Anti Aging Face Serum Headband Mask Turban Scarf Hood - 1.69 fl. oz.(해외배송)'],
    #                       tc.words)
    # tc.getTermStat()[0]
    # tc_valid.getTermStat()[0]
    # print tc.corpus
    # print tc.words#oops: only alphanumeric 'engligh" words
    # print tc.chars
    # r = tc.getLengths()
    # print r[0],r[1],r[2]#74, 68, 117
    # r = tc.getLengthsByTerm()
    # print r[0],r[1],r[2]#9, 11('for' is a stop word), 17
    # r = tc.getCharStat()
    # print r[0][0], r[1][0]
    # r = tc.getSpecialCharStat()
    # print r[0][0], r[1][0], r[0][1], r[1][1], r[0][2], r[1][2] # (3,?), (4,?), (0,?)
    # r = tc.hasSpecialChar()
    # print r[0], r[1], r[2] #True True False
    # r = tc.getUpperCharStat()
    # print r[0][0], r[1][0], r[0][1], r[1][1], r[0][2], r[1][2] # (11,?), (16,?), (19,?)
    # tc.getTermStat()
    ## tc.getCfIdf()
    # r = tc.getTfIdF()
    # print r[0],r[1],r[2]
    # r = tc.getPosStat()
    # [{'CD': 2, 'JJ': 1, 'NNP': 8, 'PUNCT': 2},
    #  {'CD': 1, 'IN': 1, 'NNP': 9, 'PUNCT': 2},
    #  {'NNP': 17}]
    # print r, '\n', r[Pos.AdjectiveSuperlative], r[Pos.ProperNounSingular] # ?, [0,0,0], [8,9,17]
    # r = tc.getNounStat()#[8,9,17]
    # print r
    # r = tc.getVerbStat()#[0,0,0]
    # print r
    # r = tc.getAdjectiveStat()#[1,0,0]
    # print r
    # r = tc.getNumberStat()
    # print r[0], r[1], r[2] # 2, 1, 0
    # r = tc.hasNumber()
    # print r[0], r[1], r[2] # True, True, False
    # r = tc.getCharStatByLanguage()
    # print r[UnicodeBlock.CJKUnifiedIdeographs][0][0], r[UnicodeBlock.CJKUnifiedIdeographs][1][0], r[UnicodeBlock.CJKUnifiedIdeographs][0][1], r[UnicodeBlock.CJKUnifiedIdeographs][1][1]# (6,?), (0,0.)
    # r = tc.getNonEnglishCharStat()
    # print r[0][0], r[1][0], r[0][1], r[1][1], r[0][2], r[1][2] # (6,?), (0,0.), (0,0.)
    ## tc.getTermStatByLanguage()
    ## tc.getNonEnglishTermStat()
    # r = tc.getColorStat()
    # print r[0][0], r[1][0], r[0][1], r[1][1], r[0][2], r[1][2] # (1,?), (1,?), (0,?)
    # r = tc.getBrandStat([u'Vanker', u'Adidas'])#0, 0, 1
    # print r

    # features = [tc.getLengths()] + \
    #            [tc.getLengthsByTerm()] + \
    #             tc.getCharStat()[0] + \
    #            [tc.getSpecialCharStat()[0]] + \
    #            [tc.hasSpecialChar()] + \
    #            [tc.getUpperCharStat()[0]] + \
    #             tc.getTermStat()[0] + \
    #             tc.getTfIdF() + \
    #            [tc.getNounStat()] + \
    #            [tc.getVerbStat()] + \
    #            [tc.getAdjectiveStat()] + \
    #            [tc.hasNumber()] + \
    #            [tc.getNumberStat()] + \
    #            [tc.hasTamilChar()] + \
    #            [tc.hasChineseChar()] + \
    #            [tc.getNonEnglishCharStat()[0]]  + \
    #            [tc.getColorStat()[0]] + \
    #            [tc.getBrandStat([u'Vanker', u'Adidas'])] + \
    # features = [tc.getSyllableStat()] + \
    #            [tc.getPolysyllabStat()] + \
    #            [tc.getAvgLetterPerWord()] + \
    #            [tc.getAvgSentencePerWord()] + \
    #            [tc.getAvgSyllablePerWord()] + \
    #            [tc.getColemanLiauIndex()] + \
    #            [tc.getDaleChallReadabilityScore()] + \
    #            [tc.getAutomatedReadabilityIndex()] + \
    #            [tc.getDifficultWordsStat()] + \
    #            [tc.getFleschReadingEase()] + \
    #            [tc.getFleschKincaidGrade()] + \
    #            [tc.getGunningFog()] + \
    #            [tc.getLexiconStat()] + \
    #            [tc.getLinsearWriteFormula()] + \
    #            [tc.getSmogIndex()] + \
    #            [tc.getTextStandardLevel()]
    #
    # import numpy as np
    # r = np.asarray(features).transpose()
    # print r.shape


