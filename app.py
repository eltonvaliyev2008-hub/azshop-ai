from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

with open('model/decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    yeni = pd.DataFrame({
        'Yaş':             [data['yas']],
        'Şəhər':           [data['sehir']],
        'Kateqoriya':      [data['kateqoriya']],
        'Üzvlük_Növü':     [data['uzvluk']],
        'Ödəniş_Üsulu':    [data['odenis']],
        'Trafik_Mənbəyi':  [data['trafik']],
        'Məbləğ_AZN':      [data['meblег']],
        'Əvvəlki_Alışlar': [data['evvelki']],
        'Müştəri_Rəyi':    [data['reyi']],
        'Saytda_Vaxt_Dəq': [data['saytda']]
    })
    proqnoz = model.predict(yeni)[0]
    ehtimal = model.predict_proba(yeni)[0]
    return jsonify({
        'netice': int(proqnoz),
        'alma_ehtimali': round(float(ehtimal[1]) * 100, 1),
        'almama_ehtimali': round(float(ehtimal[0]) * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)