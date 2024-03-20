import re
from newspaper import Article
import flask
from flask import request, Flask,render_template
import urllib
from flask_cors import CORS

app=Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    import pickle
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # with open('model1.pkl', 'rb') as file:
    #     model1 = pickle.load(file)
    print('above pickle')
    model1=pickle.load((open('model1.pkl', 'rb')))
    print('below pickle')
    tokenizer=Tokenizer()
    url=request.get_data(as_text=True)[5:]
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
    return render_template('index.html', prediction_text='The news is "{}"'.format(a))



if __name__ == '__main__':
    app.run(debug=True)
