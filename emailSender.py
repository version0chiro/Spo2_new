# Python code to illustrate Sending mail from 
# your Gmail account 
import smtplib 

def send_mail(mailID):
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login("adamjensenroxx@gmail.com", "Sachin@123") 

    # message to be sent 
    message = "We have obtained high temperature being recorded"

    # sending the mail 
    s.sendmail("adamjensenroxx@gmail.com", mailID, message) 

    # terminating the session 
    s.quit() 
