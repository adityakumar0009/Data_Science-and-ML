from flask import Flask,render_template,url_for,request
import joblib
model = joblib.load('linear_regression_model.lb')
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_data')
def user_data():
    return render_template('user_data_.html')

@app.route('/predict',methods=['Get','Post'])
def predict():
    if request.method == 'Post':
        brand_name = int(request.form['brand_name'])
        Kms_Driven = int(request.form['Kms_Driven'])
        owner = int(request.form['owner'])
        age = int(request.form['age'])
        power = int(request.form['power'])
        
        user_data = [[owner,brand_name,Kms_Driven,age,power]]
        pred = model.predict(user_data)[0][0] #[25141.231]
        prediction = str(round(pred,2))
        return render_template('user_data.html',prediction_text = prediction)

if __name__ == "__main__":
    app.run(debug=True)

    