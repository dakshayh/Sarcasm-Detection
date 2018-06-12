
# coding: utf-8

# In[53]:

'''
Sarcasm Detection:

Refer this web page: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d

https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b

and part 3 as well
'''


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding


# In[54]:


import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from numpy import array
from numpy import asarray
from numpy import zeros

from sklearn.manifold import TSNE


# In[55]:


data = pd.read_csv('/home/ubuntu/Desktop/Internship/SarcasmDetection/data/sample_data.csv', sep='\t',header=None)
X = data.iloc[:,1].values
Xdata = []

labels = data.iloc[:,0].values

for i in range(len(X)):
    Xdata.append(str(X[i]))


# In[56]:


t = Tokenizer()
t.fit_on_texts(Xdata)
vocab_size = len(t.word_index) + 1


# In[57]:


encoded_docs = t.texts_to_sequences(Xdata)
print(encoded_docs)


# In[58]:


print(Xdata)


# In[59]:


max_length = max([len(encoded_docs[i]) for i in range (0,len(encoded_docs))])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# In[60]:


embeddings_index = dict()
f = open('/home/ubuntu/Desktop/Internship/SarcasmDetection/glove/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[61]:


embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# In[62]:


## create model
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[63]:


print(model_glove.summary())


# In[64]:


model_glove.fit(padded_docs, np.array(labels),validation_split=0.4,epochs = 50)   


# In[65]:


loss, accuracy = model_glove.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[66]:


model_glove.save('glove_LSTM.h5')


# In[67]:


from keras.models import load_model

model=load_model('glove_LSTM.h5')


# In[68]:


data = pd.read_csv('/home/ubuntu/Desktop/Internship/SarcasmDetection/data/sample_test_data.csv', sep='\t',header=None)
X = data.iloc[:,1].values
Xtest = []

for i in range(len(X)):
    Xtest.append(str(X[i]))


# In[69]:


print(Xtest)


# In[70]:


encoded_test = t.texts_to_sequences(Xtest)


# In[71]:


padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_test)


# In[72]:


ytest=model.predict_classes(padded_test)


# In[73]:


print(ytest)

