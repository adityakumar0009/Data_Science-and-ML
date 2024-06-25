from flask import Flask,render_template,url_for,request
import joblib
model = joblib.load('BNB_Model.lb')
countvectorizer = joblib.load('countvectorizer.lb')
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        email_message = str(request.form['email_message'])
        email = [email_message]
        transform_email = countvectorizer.transform(email)
        print(transform_email.shape)
        prediction = str(model.predict(transform_email)[0])
        print(prediction)
        dt = {'0':'ham','1':'spam'}
        return dt[prediction]
        # return email_message
if __name__ =="__main__":
    app.run(debug=True)