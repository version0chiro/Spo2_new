import bcrypt

# passwd = bytes('sachin', encoding='utf-8')


# salt = bcrypt.gensalt()
# hashed = bcrypt.hashpw(passwd, salt)

# print(salt)
# print(hashed)
# name='sachin'
# p=bcrypt.hashpw(bytes(name, encoding='utf-8'),salt)

def checkPassword(password):
    passwd=b'pyqt543'
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passwd, salt)
    
    password_bcrypt=bcrypt.hashpw(bytes(password, encoding='utf-8'),salt)
    if password_bcrypt==hashed:
        return 1
    else:
        return 0
