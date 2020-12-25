import bcrypt
import os
from datetime import date
import os


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
            expire_date=pickle.load( open( "date.p", "rb" ))
            today = date.today()
            difftime = expire_date - today
            if difftime > datetime.timedelta(days=0):
                if os.path.isfile('salt.p'):
                    salt=pickle.load( open( "salt.p", "rb" ))
                    userHashed = bcrypt.hashpw(password, salt)
                    p = open("bycrpt.txt", "r")
                    txt=p.read()
                    if str(txt)==str(userHashed):
                        return 1
                    else:
                        return 0
        else:
            return 2
    except FileNotFoundError:
        return 2


