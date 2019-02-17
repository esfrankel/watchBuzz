import os
import googleapiclient.discovery
import pandas as pd
import numpy as np
from scipy.stats import skew
import librosa
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flask import Flask
from flask import send_from_directory
from flask import request
from werkzeug import secure_filename
import base64
import re

scaler = joblib.load("./watchBuzz/scaler.save")
model = joblib.load('./watchBuzz/model.joblib')
pca = joblib.load("./watchBuzz/pca.save")

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_url_path='')
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return send_from_directory('static','index.html')

    @app.route('/audio', methods=['POST'])
    def audio():
        SAMPLE_RATE = 44100

        # def decode_base64(data, altchars=b'+/'):
        #     """Decode base64, padding being optional.
        #
        #     :param data: Base64 data as an ASCII byte string
        #     :returns: The decoded byte string.
        #
        #     """
        #     data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
        #     missing_padding = len(data) % 4
        #     if missing_padding:
        #         data += b'='* (4 - missing_padding)
        #     return base64.b64decode(data, altchars)

        f = request.files['file']

        # data = request.get_json(force=True)['data']
        # with open("audio.wav", 'wb') as file:
        #     file.write(base64.b64decode(data))


        # print(decode_base64(data['data']+b'==='))

        # print(request.files)
        # print("before")
        print("Before")
        print("After")
        # print("after")
        f.save("./watchBuzz/" + secure_filename(f.filename))

        def get_mfcc(path):
            data, _ = librosa.core.load(path, sr = SAMPLE_RATE)
            assert _ == SAMPLE_RATE
            try:
                ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
                ft2 = librosa.feature.zero_crossing_rate(data)[0]
                ft3 = librosa.feature.spectral_rolloff(data)[0]
                ft4 = librosa.feature.spectral_centroid(data)[0]
                ft5 = librosa.feature.spectral_contrast(data)[0]
                ft6 = librosa.feature.spectral_bandwidth(data)[0]
                ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
                ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
                ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
                ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
                ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
                ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
                return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
            except:
                print('bad file')
                return pd.Series([0]*210)

        def pca_scaler(row):
            scaled = scaler.transform(row)
            scaled_pca = pca.transform(scaled)
            return scaled_pca

        def totalizer(path):
            print("mememees")
            train = pd.DataFrame(get_mfcc(path)).T
            ret = pca_scaler(train)
            return ret

        audioMap = ['Air Conditioner','Car Horn','Children Playing','Dog Barking','Drilling','Engine Idling','Gun Shot','Jack Hammer','Siren', 'Street Music', '']

        def convert_to_labels(preds, i2c, k=1):
            ans = 0
            id = 10
            count = 0
            for p in preds:
                if p >= 0.55:
                    ans = p
                    id = count
                count += 1

            return i2c[id]

        def predictTest(instance):
            print("helloooooo")
            print(convert_to_labels(model.predict_proba(instance)[0], audioMap))
            return convert_to_labels(model.predict_proba(instance)[0], audioMap)

        x = "./watchBuzz/audio.wav" #test_audio/Test/2519.wav

        holder = predictTest(totalizer(x))
        return holder;

    return app
