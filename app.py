import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pickle
from newspaper import Article
import flask
from flask import request, Flask,render_template
import urllib
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

app=Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    model1= pickle.load(open('model1.pkl','rb'))
    tokenizer=Tokenizer()
    url=request.get_data(as_text=True)
    url=urllib.parse.unquote(url)
    article=Article(str(url))
    article.download()
    article.parse()
    x=article.title+" "+ article.text
    x=x.lower()
    x=re.sub(r'[^\w\s]',"",x)
    x=re.sub(r'\d',"",x)
    x=''.join(x)
    a=[]
    a.append(x)
    a=tokenizer.texts_to_sequences(a)
    a=pad_sequences(a,maxlen=1000)
    r=model1.predict(a)
    return render_template('index.html', prediction_text='The news is "{}"'.format(r[0][0]))

if __name__ == '__main__':
    app.run(debug=True)


# tokenizer=Tokenizer()


# model = pickle.load(open('model.pkl', 'rb'))
# url='https://www.bbc.com/news/world-asia-65581169'
# article=Article(url)
# article.download()
# article.parse()
# x=article.title+" "+ article.text
# print(x)
# x=x.lower()
# x=re.sub(r'[^\w\s]',"",x)
# x=re.sub(r'\d',"",x)
# x=''.join(x)
# a=[]
# a.append(x)
# a=tokenizer.texts_to_sequences(a)
# a=pad_sequences(a,maxlen=1000)
# r=model.predict(a)
# print(r)