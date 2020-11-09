from flask import Flask, jsonify, request
from parallel import ReSolver
from skew_correction import SkewCorrection
from PIL import Image
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['POST'])
def index():
    if request.method == 'POST':
        # request_data = request.form['text']
        # print(request_data )
        img = request.files.get('image', '')
        img = Image.open(img)
        img = np.array(img)
        obj = ReSolver(img)
        img = obj.orifice()
        obj = SkewCorrection(img)
        img = obj.main()
        im = Image.fromarray(img)
        im.save("processed.jpg")
        return jsonify({"status": "processed_image"})
    else:
        return jsonify({"status": "processed_image"})

app.run()