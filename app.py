from flask import app, Flask, flash, redirect, render_template, request, url_for
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "temp12345"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

#saved_eye_model = load_model('models/eye_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image_api():
    if 'uploaded_file' not in request.files:
        flash('Image is not provided!')
        return redirect(request.url)

    uploaded_file = request.files['uploaded_file']
    if uploaded_file.filename == '':
        flash('No image is selected for uploading!')
        return redirect(request.url)

    uploaded_filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(
        app.config['UPLOAD_FOLDER'], uploaded_filename))

    flash('Image was successfully uploaded and desplayed below:')

    return render_template('index.html', filename=uploaded_filename)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
