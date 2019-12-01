
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

pickleFile = open('pickleDetect','rb')
t = pickle.load(pickleFile)
max_length = pickle.load(pickleFile)
padded_docs = pickle.load(pickleFile)
embedding_matrix = pickle.load(pickleFile)
labels = pickle.load(pickleFile)
vocab_size = pickle.load(pickleFile)

print(max_length)

## create model
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
model_glove.add(Bidirectional(LSTM(100)))
model_glove.add(Dense(30,activation = "relu"))
model_glove.add(Dense(15,activation = "relu"))
model_glove.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit the model
model.fit(padded_docs, labels, validation_split=0.33, batch_size=32 ,epochs=20, callbacks=callbacks_list, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
