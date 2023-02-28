import pickle
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import tensorflow as tf

from os import path
import urllib.request as request
from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


app = Flask(__name__)


def download_url(url, save_path):
    with request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())


saved_model = ['https://github.com/sankirnajoshi/sentiment-app/raw/master/model/model.h5',
               'https://raw.githubusercontent.com/sankirnajoshi/sentiment-app/master/model/model.json',
               'https://github.com/sankirnajoshi/sentiment-app/raw/master/model/tokenizer.pickle'
               ]

if not path.exists('./model'):
    os.makedirs('./model')

if not path.exists('./model/model.h5'):
    download_url(saved_model[0], './model/model.h5')
if not path.exists('./model/model.json'):
    download_url(saved_model[1], './model/model.json')
if not path.exists('./model/tokenizer.pickle'):
    download_url(saved_model[2], './model/tokenizer.pickle')


with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load json and create model
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./model/model.h5")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    response = data['text']
    response = tokenizer.texts_to_sequences([response])
    response = sequence.pad_sequences(response, maxlen=48)
    probs = np.around(model.predict(response), decimals=2)
    pred = np.argmax(probs)
    if pred == 0:
        tag = 'Very Negative'
        tag_prob = probs[0, 0]
        sent_prob = np.sum(probs[0, :2])
    elif pred == 1:
        tag = 'Negative'
        tag_prob = probs[0, 1]
        sent_prob = np.sum(probs[0, :2])
    elif pred == 2:
        tag = 'Neutral'
        tag_prob = probs[0, 2]
        sent_prob = probs[0, 2]
    elif pred == 3:
        tag = 'Positive'
        tag_prob = probs[0, 3]
        sent_prob = np.sum(probs[0, 3:])
    elif pred == 4:
        tag = 'Very Positive'
        tag_prob = probs[0, 4]
        sent_prob = np.sum(probs[0, 3:])
    return jsonify({'tag': str(tag), 'tag_prob': str(tag_prob), 'sent_prob': str(sent_prob)})


if __name__ == '__main__':
    app.run(debug=True)
