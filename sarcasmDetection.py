
# coding: utf-8

# In[53]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding


import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from numpy import array
from numpy import asarray
from numpy import zeros

from sklearn.manifold import TSNE


data = pd.read_csv('/home/ubuntu/Desktop/Internship/SarcasmDetection/data/sample_data.csv', sep='\t',header=None)
X = data.iloc[:,1].values
Xdata = []

labels = data.iloc[:,0].values

for i in range(len(X)):
    Xdata.append(str(X[i]))


t = Tokenizer()
t.fit_on_texts(Xdata)
vocab_size = len(t.word_index) + 1


encoded_docs = t.texts_to_sequences(Xdata)
print(encoded_docs)


print(Xdata)


max_length = max([len(encoded_docs[i]) for i in range (0,len(encoded_docs))])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


embeddings_index = dict()
f = open('/home/ubuntu/Desktop/Internship/SarcasmDetection/glove/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
## create model
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_glove.summary())

model_glove.fit(padded_docs, np.array(labels),validation_split=0.4,epochs = 50)   

loss, accuracy = model_glove.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

model_glove.save('glove_LSTM.h5')

#prediction
from keras.models import load_model

model=load_model('glove_LSTM.h5')

data = pd.read_csv('/home/ubuntu/Desktop/Internship/SarcasmDetection/data/sample_test_data.csv', sep='\t',header=None)
X = data.iloc[:,1].values
Xtest = []

for i in range(len(X)):
    Xtest.append(str(X[i]))

print(Xtest)
encoded_test = t.texts_to_sequences(Xtest)
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_test)

ytest=model.predict_classes(padded_test)
print(ytest)
