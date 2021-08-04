# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:26:26 2021

@author: SHASHWAT
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2})

print(r.json())