# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:11:35 2022

@author: Eric
"""

import requests

URL = "http://127.0.0.1:5000/predict"
headers = {"Content-Type" : 'application/json'}
data = {"input" : data_in}

r = requests.get(URL, headers=headers, json=data)

r.json()

