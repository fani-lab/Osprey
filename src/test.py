import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from scipy import sparse
import numpy as np

# Preprocess the data
loader = np.load("w2v_glove.npz") #load the sparse matrix
y= pd.read_pickle('df_train_test.pkl')
y= y['tagged_msg']  #get 1 or 0 labels
X = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choose a model and train it
#model = RandomForestClassifier()
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)

# Sample test
testinput = "Hi! How are you? I just got back from walking my dog."
sentence_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
sentence_embeddings = sentence_model.encode(testinput).reshape(1, -1) #map sentence to vector
print(sentence_embeddings.shape)
print(model.predict(sentence_embeddings)) #not sure if this actually works
print('Accuracy: ', accuracy)

#y= y['tagged_msg'].value_counts()
#print(X.shape)