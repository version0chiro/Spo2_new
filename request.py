import requests



def getRequest():
    x = requests.get('http://localhost:3000/python');
    json_data = x.json()
    
    return json_data

def sendRequest():
    x = requests.get('http://localhost:3000/pythonReset');
    json_data = x.json()

def url_ok():
    try:
        r = requests.head("http://localhost:3000/python")
    except:
        r=0
    if r==0:
        return False
    else:    
        return r.status_code == 200

def upload():
    with open("excel_sheets/attendance.xlsx", "rb") as a_file:

        file_dict = {"attendance": a_file}

        response = requests.post("http://localhost:3000/checkFile", files=file_dict)

