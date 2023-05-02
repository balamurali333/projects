'''from flask import Flask,render_template,Response
from flask_wtf import FlaskForm
from wtforms import FileField , SubmitField , FloatField ,IntegerField
from wtforms.validators import InputRequired
import joblib
import numpy as np

app =  Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'


knn_from_joblib = joblib.load('model.pkl')


class UploadFileForm(FlaskForm):
    x1 = IntegerField('val1', validators=[InputRequired()])
    x2 = IntegerField('val2', validators=[InputRequired()])
    x3 = IntegerField('val3', validators=[InputRequired()])
    x4 = FloatField('val4', validators=[InputRequired()])
    x5 = FloatField('val5', validators=[InputRequired()])
    x6 = FloatField('val6', validators=[InputRequired()])
    x7 = FloatField('val7', validators=[InputRequired()])
    submit = SubmitField("Crop")


@app.route('/',methods = ["GET","POST"])
@app.route('/home',methods = ["GET","POST"])
def home():
    return render_template('home.html')


@app.route('/about',methods = ["GET","POST"])
def about():
    return render_template('about.html')

@app.route('/contact',methods = ["GET","POST"])
def contact():
    return render_template('contact.html')

@app.route('/form',methods = ["GET","POST"])
def form():
    form = UploadFileForm()
    if form.validate_on_submit():

        x1 = form.x1.data
        x2 = form.x2.data
        x3 = form.x3.data
        x4 = form.x4.data
        x5 = form.x5.data
        x6 = form.x6.data
        x7 = form.x7.data

        data = np.array([[x1,x2,x3,x4,x5,x6,x7]])
        my_prediction = knn_from_joblib.predict(data)
        final_prediction = my_prediction[0]
        final_prediction = final_prediction.capitalize()

        pa = './static/css/cropes/'+final_prediction+'.jpg'

        return render_template('sample.html',imgpath = pa,text = final_prediction)
        
    return render_template('index.html',form = form)

if __name__ == '__main__':
    app.run(debug=True)'''

# from flask import render_template,jsonify, Flask
from flask import Flask,render_template,Response, redirect, url_for, request ,jsonify
from flask_wtf import FlaskForm
from wtforms import FileField , SubmitField
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing import image
from wtforms.validators import InputRequired

import random
import os
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.applications.mobilenet import MobileNet 
# from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K

app = Flask(__name__)
# app =  Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'


CLASSES = {
  0: 'Doubtful',
  1: 'Mild',
  2: 'Moderate',
  3: 'Normal',
  4: 'Severe'

}


class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/departments')
def departments():
    return render_template('depatments.html')

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    form = UploadFileForm()
    if form.validate_on_submit():
        f = form.file.data
        directory = f.filename
        path='./static/'+f.filename
        f.save(path)
        j_file = open('modeljson.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('model_1.h5')
        img1 = image.load_img(path, target_size=(128,128))
        img1 = np.array(img1)
        img1 = img1.reshape((1,128,128,3))
        img1 = img1/255
        prediction = model.predict(img1)
        pred = np.argmax(prediction)
        # print(pred)
        disease = CLASSES[pred]
        accuracy = prediction[0][pred]
        # print(disease,accuracy)
        K.clear_session()
        return render_template('uploaded.html',imgpath = path,text = disease)
    return render_template('knee.html', form = form)

if __name__ == "__main__":
    app.run(debug=True)