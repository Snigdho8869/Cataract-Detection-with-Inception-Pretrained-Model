import os
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
from flask import jsonify


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('cataract_tl.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def traffic_predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        image = load_img(file_path, target_size=(224, 224))
        input_arr = img_to_array(image)
        input_arr = input_arr.reshape((1,) + input_arr.shape)
        preds = model.predict(input_arr)
        y_pred = np.argmax(preds, axis=1)
        
        if y_pred == 1:
            prediction = "Normal"
        else:
            prediction = "Cataract"

        response = {'signature': prediction}
        return jsonify(response)
        
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
