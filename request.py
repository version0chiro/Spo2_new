import requests

def getRequest():
    x = requests.get('http://localhost:3000/python');
    json_data = x.json()
    print(json_data)
    return json_data