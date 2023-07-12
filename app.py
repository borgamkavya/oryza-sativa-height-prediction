from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import data_processing
import csv
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    predicted_subpopulation=None
    predicted_height=None
    if request.method == 'POST':
        input_sequence = request.form.get('Sequence')
        predicted_subpopulation, predicted_height = data_processing.predict_height(str(input_sequence))
    return render_template('index.html', subpopulation=predicted_subpopulation, height=predicted_height)

@app.route('/about')
def about():
    with open('static/imputed(csv).csv', 'r') as file:
        file_content = file.readlines()
    file1_content = [row.strip().split(',') for row in file_content]
    
    with open('static/phenotype(csv).csv', 'r') as file:
        file_content = file.readlines()
    file2_content = [row.strip().split(',') for row in file_content]
     
    file_path = 'static/phylo_tree.txt'  # Update with your file path
    with open(file_path, 'r') as file:
        file_content = file.read()
    return render_template('about.html', file1_content=file1_content, file2_content=file2_content,file_content=file_content)

if __name__ == "__main__":
    data_processing.train_model()
    app.run(debug=True)
