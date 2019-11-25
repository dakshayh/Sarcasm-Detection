import pickle
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


data = pd.read_csv('cleaned_data.csv', sep='\t',header=None)
X = data.iloc[:1000,1:2].values
Xdata = []

labels = data.iloc[:1000,0:1].values

for i in range(len(X)):
    Xdata.append(str(X[i]))
    
    
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(Xdata)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(Xdata)
print(encoded_docs)
max_length = max([len(encoded_docs[i]) for i in range (0,len(encoded_docs))])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# load the whole embedding into memory
embeddings_index = dict()
f = open('/media/nikhil/5204303204301B83/cs/IIIT/classification/glove/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        

pickleFile = open('pickleDetect','wb')

pickle.dump(t,pickleFile)
pickle.dump(max_length,pickleFile)
pickle.dump(padded_docs,pickleFile)
pickle.dump(embedding_matrix, pickleFile)
pickle.dump(labels,pickleFile)
pickle.dump(vocab_size,pickleFile)

pickleFile.close()
