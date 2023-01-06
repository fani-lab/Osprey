import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from scipy import sparse
import numpy as np
from imblearn.over_sampling import RandomOverSampler
#text cleaning
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
import pickle

ROS = RandomOverSampler(sampling_strategy=1)

def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

#remove punctuation and special characters
def remove_punctuation(text):
    return re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

#remove common words that add little meaning
def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

# reducing words to their base form
def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

sentence_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

# Preprocess the data
loader = np.load("w2v_glove.npz") #load the sparse matrix
df= pd.read_pickle('df_train_test.pkl')
y= df['tagged_msg']  #get 1 or 0 labels
X = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
df['before_cleaning'] = df['text'].apply(lambda x: len(x))

#normalize text
df['text'] = df['text'].apply(lambda x: convert_to_lower(x))
df['text'] = df['text'].apply(lambda x: remove_numbers(x))
df['text'] = df['text'].apply(lambda x: remove_punctuation(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: remove_extra_white_spaces(x))
df['text'] = df['text'].apply(lambda x: lemmatizing(x))

df['after_cleaning'] = df['text'].apply(lambda x: len(x))

print(df["text"].head(25))

features = sparse.csr_matrix((0,  len(df))).transpose()
sentence_embeddings = sentence_model.encode(df['text'].values)

X = sparse.csr_matrix(sparse.hstack((features, sentence_embeddings)))

X,y = ROS.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choose a model and train it
#model = RandomForestClassifier()
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)

# Sample test
testinput = "Hi! How are you? I just got back from walking my dog."
sentence_embeddings = sentence_model.encode(testinput).reshape(1, -1) #map sentence to vector
print(sentence_embeddings.shape)
print(model.predict(sentence_embeddings)) #not sure if this actually works
print('Accuracy: ', accuracy)

#y= y['tagged_msg'].value_counts()
print(y.value_counts())
pickle.dump(model, open("model.joblib", 'wb'))