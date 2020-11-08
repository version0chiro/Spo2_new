import requests

def getRequest():
    x = requests.get('http://localhost:3000/python');
    json_data = x.json()
    
    return json_data

def sendRequest():
    x = requests.get('http://localhost:3000/pythonReset');
    json_data = x.json()