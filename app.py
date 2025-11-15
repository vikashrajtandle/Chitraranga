from flask import Flask, render_template, redirect, request
import tensorflow as tf
import os
from PIL import Image
import numpy as np

#Model loading
model_path = "UNet_chitraranga_model.keras" 

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

Model = tf.keras.models.load_model(model_path, custom_objects={'psnr': psnr})

#Ensuring that the 'static' folder exists
os.makedirs('static', exist_ok=True)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        file = request.files['image-upload']  

        if not file:
            return "No file Uploaded",400
        
        #Preprocessing the input image
        img = Image.open(file).convert("L") # grayscale input
        input_width,input_height = img.size #storing the original input image size
        img = img.resize((128,128))         # resize to training size
        img_array = np.array(img)           # convert the image into an array of pixles
        img_array = img_array/ 255.0        # normalize
        img_array = np.expand_dims(img_array, axis=(0,-1))  #(1,128,128,1) 1-input of (128x128) image with 1 channel

        #Predict colorize
        pred = Model.predict(img_array)[0]
        pred = (pred*255).astype(np.uint8) #Gray to color and float to int

        #Result image
        result_img = Image.fromarray(pred)
        result_img = result_img.resize((input_width,input_height))
        result_path = "static/result.png"
        result_img.save(result_path)

        return render_template('base-result.html', result = result_path) 
    return render_template('base-upload.html') 

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)