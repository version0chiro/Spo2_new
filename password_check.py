import bcrypt
import os
from datetime import date
import os
import datetime

import pickle

def checkPassword(password):
    passwd=b'pyqt543'
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passwd, salt)
    
    password_bcrypt=bcrypt.hashpw(bytes(password, encoding='utf-8'),salt)
    if password_bcrypt==hashed:
        return 1
    else:
        return 0
    
def check_Password(password):
    try:
        if os.path.isfile("password/date.p"):
            
            expire_date=pickle.load( open( "password/date.p", "rb" ))
            today = date.today()
            difftime = expire_date - today
            
            if difftime > datetime.timedelta(days=0):
                if os.path.isfile('password/salt.p'):
                    salt=pickle.load( open( "password/salt.p", "rb" ))
                    userHashed = bcrypt.hashpw(bytes(password, encoding='utf-8'), salt)
                    p = open("password/bycrpt.txt", "r")
                    txt=p.read()
                    if str(txt)==str(userHashed):
                        return [1,difftime.days]
                    else:
                        return [0,difftime.days]
        else:
            return [2,0]
    except FileNotFoundError:
        return [2,0]


