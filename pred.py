from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import pickle

pickleFile = open('pickleDetect','rb')
t = pickle.load(pickleFile)
max_length = pickle.load(pickleFile)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

twt = ['I prefer the Anakin and Padme tiger scene in Ep 2']
pred_docs = t.texts_to_sequences(twt)
twt = pad_sequences(pred_docs, maxlen=max_length, padding='post')
output = loaded_model.predict_classes(twt)
print(output)
