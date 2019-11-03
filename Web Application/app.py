import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import pandas as pd
from bs4 import BeautifulSoup
import re
import pickle
import nltk
import numpy as np
from nltk.corpus import indian
#nltk.download('indian')
from nltk.tokenize import word_tokenize
import tqdm
#nltk.download('punkt')
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      #name = result['name']
      #print(name)
      text = result['Name']
      print(result['Name'])
      print(type(text))





      x = [text]#['*Veeru...* ने आप जैसे खास लोगो के भेजा है कुछ खास तोफहा ||\n यहाँ देखे आपका उपहार हमारी सौगात..... \n  \n live-wishing.club/?n=Veeru...']

      dx = np.array(x)
          
      preprocessed_reviews = []
      for sentance in dx:
          sentance = re.sub(r"http\S+", "", sentance)
          sentance = BeautifulSoup(sentance, 'lxml').get_text()
          sentance = re.sub("\n*", "", sentance).strip()
          preprocessed_reviews.append(sentance.strip())


      tokenized = []
      for line in preprocessed_reviews:
          x = nltk.tokenize.word_tokenize(line)
          tokenized.append(x)
      #print(preprocessed_reviews)     
      bad_chars = [';', ':', '!', "*",',','#','``','...','..','.','**','?','(',')','@','$','&','।','%','}','{','[',']',';']
      clean_text = []
      for line in tokenized:
          text = ""
          for word in line:
              if word not in bad_chars:
                  text = text+" "+word
          clean_text.append(text)
          
      tokenized = []
      for line in clean_text:
          x = nltk.tokenize.word_tokenize(line)
          tokenized.append(x)
          
      clean_text = []
      for line in tokenized:
          text = ""
          for word in line:
              for i in bad_chars: 
                  word = word.replace(i, '')
              text = text+' '+word
          clean_text.append(text)
          
      final_text = []
      for line in clean_text:
          final_text.append(line.strip()) 
          
      clean_text = []
      for line in final_text:
          text = ''
          for word in line.split():
              if len(word)>1:
                  text = text+' '+word
          clean_text.append(text)
          
      text_data = []
      for line in clean_text:
          v = ''
          for word in line.split():
                  v = v+' '+word
          text_data.append(v)
          
      final_text = []
      for line in text_data:
          final_text.append(line.strip())

      result1 = {}  #it will store all the results that need to be shown on the final html page
      result1['Message'] = final_text[0]

      
          
      # Predicting whether the message is political or not
      tfidf = pickle.load(open('TFIDF_Political.pickle', 'rb'))  #loading the pretrained model for text featurization
      features = tfidf.transform(final_text)                    # converting the features into vector representation
      NaiveBayes = pickle.load(open('Naive_Bayes_Political.pickle', 'rb'))   #loading the machine learning model for prediction
      prediction = NaiveBayes.predict(features)

      if prediction[0]==0:
         typ = 'Non-Political'
      else:
         typ = 'Political'

      result1['Category'] = typ
          
      # predicting the category of the message i.e. Offensive, Adversisenment, Spam or Normal Message      
      tfidf = pickle.load(open('TFIDF.pickle', 'rb'))  #loading the pretrained model for text featurization
      features = tfidf.transform(final_text)               # converting the features into vector representation
      NaiveBayes = pickle.load(open('Naive_Bayes.pickle', 'rb')) #loading the machine learning model for prediction
      prediction = NaiveBayes.predict(features)
      print(prediction[0])

      
      
      
      if(prediction[0]==0):
         typ = 'Spam or 0ffensive or Advertisement'
      else:
         typ = 'Normal Message'
      result1['Type of Message'] = typ
      print(result1)
      
      return render_template("result.html",result = result1)

if __name__ == '__main__':
   app.run(debug = True)
