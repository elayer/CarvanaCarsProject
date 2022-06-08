# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:30:13 2022

@author: Eric
"""

import flask
from flask import Flask, jsonify, request, render_template, request, redirect, url_for
import json

import numpy as np
import pickle

#Mehod to load the model for the app
def load_models():
    file_name = 'models/model_file.p'
    
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
        
    return model

#Method to load select scalers for the app to properly scale the data prior to prediction
def load_scaler(scaler_name):
    file_name = 'models/scaler.pkl'
    
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        scaler = data['scaler']
    return scaler


#Method to make a prediction using a sample of data
app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict(sample):
    
    model = load_models()
    prediction = model.predict(sample) 
    
    #return response, 200
    
    return prediction

#Method to render the home page containing the form to submit a data sample
@app.route('/')
def home():
    return render_template('index.html')

#Method to create and format a data sample and call the result page to display the resulting prediction
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        
        #Loading scaler
        scaler = load_scaler('models/scaler.pkl')
        
        
        #Take in the submitted form's data and transform appropriately
        initial_data = request.form.to_dict()
        initial_data = list(initial_data.values())
        
        initial_data = list(map(float, initial_data))
        
        initial_data = np.array(initial_data).reshape(1, -1)
        
        initial_data_scaled = scaler.transform(initial_data)
        
        
        print(initial_data_scaled)
        
        #Since we have log prices, we have to use np.exp to return the value to the correct scale
        pred = '$'+str(round(float(np.exp(predict(initial_data_scaled))), 2))
        
        
        return render_template('result.html', prediction = pred)
        

if __name__ == '__main__':
    app.run(debug=True)