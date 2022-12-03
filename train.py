import pandas as pd
import os
import pickle
import sys

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils.preprocessor import Preprocess

if __name__ == '__main__':

    data = Preprocess().dataset
    data = data.dropna()
    x, y = data['text'], data['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if os.path.exists('models/vectorizer.pkl'):
        print("\nLoading Vectorizer...")

        with open('models/vectorizer.pkl', 'rb') as handle:
            vectorizer = pickle.load(handle)

        print('[Loading Completed]\n')
    else:
        print("\Building Vectorizer...")

        vectorizer = TfidfVectorizer()
        vectorizer.fit(x_train)

        with open('models/vectorizer.pkl', 'wb') as handle:
            pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('[Vectorizer Completed]\n')

    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    print('Train set:', x_train.shape,  y_train.shape)
    print('Test set:', x_test.shape,  y_test.shape)

    print('Training SVM LinearSVC Model...')
    svm = SVC(verbose=True, kernel='rbf', random_state=42)
    svm.fit(x_train, y_train)
    print('[ Training Completed ]\n')

    print('Saving SVM LinearSVC Model...')
    with open('models/svm.pkl', 'wb') as handle:
        pickle.dump(svm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('[Saving Completed]\n')

    linear_svc_score = svm.score(x_test, y_test)
    print(f'Score: {linear_svc_score*100}\n')

    y_pred = svm.predict(x_test)

    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred))

