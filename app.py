import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

with open('models/vectorizer.pkl', 'rb') as handle:
    vectorizer = pickle.load(handle)

with open('models/svm.pkl', 'rb') as handle:
    model = pickle.load(handle)

stop_words = StopWordRemoverFactory().get_stop_words()
stemmer = StemmerFactory().create_stemmer()
result = ['NETRAL', 'POSITIF', 'NEGATIF']


def predict(text):
    result = text.lower()
    result = re.sub('\n', ' ', result)
    result = re.sub(r'@\w+', '', result)
    result = re.sub(r'http\S+', '', result)
    result = result.translate(str.maketrans('', '', string.punctuation))
    result = re.sub("'", '', result)
    result = re.sub(r'\d+', '', result)
    result = ' '.join([word for word in result.split() if word not in stop_words])
    result = stemmer.stem(result.strip())
    result = vectorizer.transform([result])
    result = model.predict(result[0])

    return result


st.write('# Sentiment Analysis Bahasa')
st.write('Aplikasi untuk prediksi sentiment suatu text')

st.write('## Dataset')
st.write(pd.read_csv('data/twitter.csv'))

st.write('## Predict')
text = st.text_input('Text', placeholder='Masukkan text')

if text:
    predicted = predict(text)
    st.write('## Result')
    st.write(f'Sentiment text anda adalah **{result[int(predicted)]}**')
