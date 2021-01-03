import bcrypt
import os
from datetime import date
import os
import datetime
import pymongo
from getmac import get_mac_address as gma
import pickle
import re, uuid

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
    myclient = pymongo.MongoClient("mongodb://admin-sachin:Sachin123@cluster0-shard-00-00.pf7ee.mongodb.net:27017,cluster0-shard-00-01.pf7ee.mongodb.net:27017,cluster0-shard-00-02.pf7ee.mongodb.net:27017/spo2?ssl=true&replicaSet=atlas-x3z4ou-shard-0&authSource=admin&retryWrites=true&w=majority")
    mydb = myclient["spo2"]
    mycol = mydb["identities"]
    MAC = str(':'.join(re.findall('..', '%012x' % uuid.getnode())))
    # MAC = str(gma())
    
    query = {"MAC": MAC}
    try:
        search = mycol.find_one(query)
        activationFromServer = search["ActivationKey"]
        dateFromServer = search["created_at"].date()
        
    except:
        return [3,0]
  
    print("found in mongoDB")
  
    try:
        today = date.today()
        difftime =  today - dateFromServer 
        difftime = difftime.days
        print(difftime)
        if difftime > 0:
            if str(activationFromServer)==str(password):
                return [1,difftime]
            else:
                return [0,difftime]
        else:
            return [2,0]
    except FileNotFoundError:
        return [2,0]


