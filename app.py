from flask import Flask,render_template,request
import pickle
import string
import nltk
import catboost
# from nltk import stopwords,PorterStemmer
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import PorterStemmer
ps=PorterStemmer()
model=pickle.load(open('model_cat.pkl','rb'))
tfidf=pickle.load(open('vectorizer.pkl','rb')) 

app=Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/sentiment",methods=['POST',"GET"])
def sentiment():
    user_input =request.form.get("user_input")
    print(user_input)
    user_input=user_input.lower()
    text= nltk.word_tokenize(user_input)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text: 
        y.append(ps.stem(i))
        output="".join(y)
    tranform_sms=output
    vector_input=tfidf.transform([tranform_sms])
    result=model.predict(vector_input)
    return render_template("index.html",result=result)
    print(result)
if __name__=="__main__":
    app.run(debug=True)
