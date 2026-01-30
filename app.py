from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from utils import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/pneumonia_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)[0][0]

    result = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL"
    confidence = round(prediction * 100, 2)

    return render_template(
        'result.html',
        result=result,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)

