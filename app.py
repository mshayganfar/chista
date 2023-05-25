from flask import app, Flask, flash, redirect, render_template, request, url_for
import os
from process_image import Beauty
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "temp12345"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


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

    beauty = Beauty()
    predictions = beauty.classify_image_category(
        app.config['UPLOAD_FOLDER'], uploaded_filename)
    predictions.sort(key=lambda x: x[0], reverse=True)
    beauty_flag = False
    possibilities = []
    for pred_tuple in predictions:
        if pred_tuple[0] > 0.50:
            beauty_flag = True
            possibilities.append(pred_tuple[1])
            print(pred_tuple[0], pred_tuple[1])

    if beauty_flag == False:
        print(predictions)
        return render_template('index.html', filename=uploaded_filename, predition_result="Not Beauty!")
    else:
        return render_template('index.html', filename=uploaded_filename, predition_result="Beauty!", possibilities=possibilities)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
