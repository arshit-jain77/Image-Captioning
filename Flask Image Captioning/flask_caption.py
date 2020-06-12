import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename, SharedDataMiddleware
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import sys
from generate_caption import gen_caption

UPLOADER_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg','jpeg'])

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

def my_random_string(string_length=10):
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOADER_FOLDER'] = UPLOADER_FOLDER

@app.route("/")
def template_test():
    return render_template('index_caption.html',image='uploads/58368365_03ed3e5bdf.jpg') #, label='', imagesource='uploads/template.jpg')

# for static
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("static")
    if request.method == 'POST':
        print('static inside post')

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOADER_FOLDER'], filename)
            file.save(file_path)

            status = gen_caption(file_path)
            # print(status)
            os.rename(file_path, os.path.join(app.config['UPLOADER_FOLDER'], filename))
            return render_template('index_caption.html',image='uploads/' + filename,status = status)




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADER_FOLDER'],
                               filename)


app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOADER_FOLDER']
})

if __name__ == "__main__":
    app.debug = False
    app.run()







