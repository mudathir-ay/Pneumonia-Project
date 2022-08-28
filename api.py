import os
import sys
import glob
import re
import numpy as np
import h5py
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
MODEL_PATH ="model2.h5"
model = load_model(MODEL_PATH)

from flask import Flask 
from flask import request
from flask import render_template 
from werkzeug.utils import secure_filename
def model_predict(img_path, model):
  img = image.load_img(img_path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  pred=np.ndarray.item(np.argmax(preds, axis=1))
  if pred==0:
     preds="The Person is Normal"
  else:
     preds="The Person is Infected With Pneumonia"
  return preds

app=Flask(__name__)
base_path=os.path.dirname(__file__)+"/static"
print(base_path)

@app.route("/",methods=["GET","POST"])
def upload_predict():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            file_path=os.path.join(base_path,image_file.filename)
            image_file.save(file_path)
            prediction=model_predict(file_path,model)
            return render_template("index.html", prediction=prediction,image_loc=image_file.filename)
    return render_template("index.html", prediction=0,image_loc=None)

if __name__=="__main__":
    app.run(port=12000,debug=True)
