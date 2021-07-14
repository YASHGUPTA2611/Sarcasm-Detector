from flask import Flask, render_template, request,redirect,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('Sarcasm_model.h5', compile = False)

app = Flask(__name__)


@app.route('/')
def Welcome():
    return render_template('input.html')
    


@app.route('/predict', methods=['POST'])
def predict():
    ans = ""
    text = request.form['text']
    textlen = 30
    onehot = [one_hot(text, textlen)]
    result = pad_sequences(onehot, padding='pre', maxlen=textlen)
    prediction = model.predict(result)
    prediction = prediction[0]
        
    if prediction>0.5:
        ans="Sarcastic"
    else:
        ans="Not Sarcastic"        
    return render_template('output.html', text=text, ans=ans)


if __name__ == "__main__":
    app.run(debug=True)