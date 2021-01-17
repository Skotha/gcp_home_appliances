from flask import Flask, request, jsonify, render_template, send_file, url_for, send_from_directory, abort, redirect, Response
from werkzeug.utils import secure_filename
import os
#from google.cloud import storage
import json
import tensorflow as tf
import pandas as pd
from m2 import go
from time import sleep
# ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_EXTENSIONS

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
app.config["CLIENT_CSV"] = UPLOAD_FOLDER
filename = 'file.csv'
path = ''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/json')
def json_fn():
    global path
    if path != '':
        df = go(path)
        data = json.loads(df.to_json(orient='records').replace('\/', '/'))
        d = [[],[],[],[],[],[],[],[]]
        for i in data:
            for j, value in enumerate(i.values()):
                d[j].append(value)
        return Response(json.dumps(d, default=str, sort_keys=False, indent=4), content_type="application/json")


@app.route('/csv')
def csv_fn():
    global path
    print(path)
    if path != '':
        p = os.path.join(app.config['CLIENT_CSV'],'result.csv')
        
        return send_file(p,
                     mimetype='text/csv',
                     attachment_filename='result.csv',
                     as_attachment=True)
        #return send_from_directory(app.config['CLIENT_CSV'], filename,as_attachment=False)

@app.route('/go')
def g():
    df = go()
    data = json.loads(df.to_json(orient='table').replace('\/', '/'))
    # return jsonify(data)
    return Response(json.dumps(data, default=str, sort_keys=False, indent=4), content_type="application/json")
    # return redirect(url_for('index'))


@app.route('/', methods=['POST'])
def upload_files():
    global path
    uploaded_file = request.files['file']
    #filename = secure_filename(uploaded_file.filename)
    if filename != '':
        path = os.path.join(app.config['CLIENT_CSV'], filename)
        uploaded_file.save(path)
        return Response('file saved successfully')
        #return render_template('index.html')
        #df = go(path)
        #data = json.loads(df.to_json(orient='records').replace('\/', '/'))
        # return jsonify(data)data = json.loads(df.to_json(orient='records').replace('\/', '/'))
    # return jsonify(data)
        #return Response(json.dumps(data, default=str, sort_keys=False, indent=4), content_type="application/json")
    
        # return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
