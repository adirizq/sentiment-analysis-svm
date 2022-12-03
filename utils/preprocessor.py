import pandas as pd
import pickle
import os
import sys
import re
import string


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm


class Preprocess():
    def __init__(self,
                 dataset_path='data/twitter.csv',
                 preprocessed_dataset_path='data/preprocessed_twitter.pkl'):

        self.dataset_path = dataset_path
        self.preprocessed_dataset_path = preprocessed_dataset_path

        if os.path.exists(preprocessed_dataset_path):
            print("\nLoading Preprocessed Data...")
            self.dataset = pd.read_pickle(preprocessed_dataset_path)
            print('[Loading Completed]\n')
        else:
            print("\nPreprocessing Data...")
            self.dataset = self.load_data()
            print('[Preprocessing Completed]\n')

    def clean_tweet(self, text, stop_words, stemmer):
        result = text.lower()
        result = re.sub('\n', ' ', result)
        result = re.sub(r'@\w+', '', result)
        result = re.sub(r'http\S+', '', result)
        result = result.translate(str.maketrans('', '', string.punctuation))
        result = re.sub("'", '', result)
        result = re.sub(r'\d+', '', result)
        result = ' '.join([word for word in result.split() if word not in stop_words])
        result = stemmer.stem(result.strip())

        return result

    def load_data(self):
        dataset = pd.read_csv(self.dataset_path)
        dataset.columns = ['label', 'text']
        dataset = dataset[dataset['label'].isin([0, 1, 2])]

        dataset.dropna(inplace=True)
        dataset.drop_duplicates(subset=['text'], inplace=True)

        dataset = dataset.groupby('label').sample(n=2000, random_state=42)

        stop_words = StopWordRemoverFactory().get_stop_words()
        stemmer = StemmerFactory().create_stemmer()

        tqdm.pandas()
        dataset['text'] = dataset['text'].progress_apply(lambda x: self.clean_tweet(x, stop_words, stemmer))

        dataset.to_pickle(self.preprocessed_dataset_path)

        return dataset

    def dataset(self):
        return self.dataset
