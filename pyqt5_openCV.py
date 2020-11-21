# Copyright (c) Sachin Singh Bhadoriya 2020
# Please credit iosoft.blog if you use the information or software in it

VERSION = "SPO2 Estimation software"
from os import path
from datetime import date
from attendance import checkName
from request import getRequest,sendRequest,url_ok,upload
import imutils
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
import sys
from numpy.linalg import inv
import dlib
import imutils
import time
import skin_detector
import face_recognition
import pickle
import sys, time, threading, cv2
from HeartRate import Process
from IP_scan import get_IP
from IP_scan import get_value
from emailSender import send_mail
from string_manipulation import stringGetValue
from request import checkPing
from frontalFaceDetection import detectionFrontFace
import time
import os
try:
    from PyQt5.QtCore import Qt
    pyqt5 = True
except:
    pyqt5 = False
if pyqt5:
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
    from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
    from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
else:
    from PyQt4.QtCore import Qt, pyqtSignal, QTimer, QPoint
    from PyQt4.QtGui import QApplication, QMainWindow, QTextEdit, QLabel
    from PyQt4.QtGui import QWidget, QAction, QVBoxLayout, QHBoxLayout
    from PyQt4.QtGui import QFont, QPainter, QImage, QTextCursor
try:
    import Queue as Queue
except:
    import queue as Queue

IMG_SIZE    = 400,400          # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 1                # Scaling factor for display image
DISP_MSEC   = 50                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = True              # Flag to indicate capturing

setupFlag = True

NoScanFlag= False

recordFlag = False

frontalFlag = True

frameCount=0

globalCount=0

frontFaceCount = 0

duration=10

totalFrame = 250

Spo2Flag=0

FaceDetectionFlag=0

final_sig=[]

name=[]

boxes=[(100,250,200,150)]

spo2_set=[]

data = pickle.loads(open("updated.pickle", "rb").read())

process=Process()
hr=0
bpm=0
heartRate=0

def face_detect_and_thresh(frame):
    skinM = skin_detector.process(frame)
    skin = cv2.bitwise_and(frame, frame, mask = skinM)
    # cv2.imshow("skin2",skin)
    # cv2.waitKey(1)
    return skin,skinM


def spartialAverage(thresh,frame):
    a=list(np.argwhere(thresh>0))
    # x=[i[0] for i in a]
    # y=[i[1] for i in a]
    # p=[x,y]
    if a:
        ind_img=(np.vstack((a)))
    else:
        return 0,0,0
    sig_fin=np.zeros([np.shape(ind_img)[0],3])
    test_fin=[]
    for i in range(np.shape(ind_img)[0]):
        sig_temp=frame[ind_img[i,0],ind_img[i,1],:]
        sig_temp = sig_temp.reshape((1, 3))
        if sig_temp.any()!=0:
            sig_fin=np.concatenate((sig_fin,sig_temp))
    # print(sig_fin)
    for _ in sig_fin:
        if sum(_)>0:
            # test_fin=np.concatenate((test_fin,_))
            test_fin.append(_)
    # print("min=>")
    a= [item for item in sig_fin if sum(item)>0]
    # print(min(a, key=sum))
    min_value=sum(min(a, key=sum))
    max_value=sum(max(a, key=sum))
    # print(sum1)
    img_rgb_mean=np.nanmean(test_fin,axis=0)
    # print(img_rgb_mean)
    return img_rgb_mean,min_value,max_value

def MeanRGB(thresh,frame,last_stage,min_value,max_value):
    # cv2.imshow("threshh",thresh)
    # print(thresh)
    # print("==<>>")
    # print(img_rgb)
    # cv2.waitKey(1)
    # print(img_rgb[0])
    # thresh=thresh.reshape((1,3))
    # img_rgb_mean=np.nanmean(thresh,axis=0)
    a= [item for item in frame[0] if (sum(item)>min_value and sum(item)<max_value)]
    # print(a)
    # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
    # print(a[1:10])
    # img_temp = [item for item in img_rgb if sum(item)>764]
    # print(frame[0])
    # print(img_temp)
    # print(np.mean(a, axis=(0)))
    if a:
        # print("==>")
        # print(a)
        # print("==>")
        img_mean=np.mean(a, axis=(0))
        # print(img_mean)

        return img_mean[::-1]
    else:
        return last_stage

def preprocess(z1,z2,detrended_RGB,window_size,size_video,duration,frame):
    temp=(int(size_video/duration))
    f=frame-2

    main_R=[]
    main_B=[]
    out=[]
    for i in range(len(detrended_RGB)-f+1):

        temp_R=z1[i:i+f-1]
        temp_B=z2[i:i+f-1]
        p=[list(a) for a in zip(temp_R, temp_B)]

        out.append(p)
        # if not main_R:
        #     main_R.append(temp_R)
        # else:
        #     main_R=[main_R,temp_R]
        #
        # if not main_B:
        #     main_B.append(temp_B)
        # else:
        #     main_B=[main_B,temp_B]


    # out=[main_R,main_B]
    # print(out[0])
    return out[0]


def SPooEsitmate(final_sig,video_size,frames,seconds):
    A = 100.6
    B = 4.834
    ten=10
    z1=[item[0] for item in final_sig]
    z3=[item[2] for item in final_sig]
    SPO_pre=[]
    for _ in range(len(z1)):
        SPO_pre.append([z1[_],z3[_]])
    Spo2 = preprocess(z1,z3,SPO_pre,ten,video_size,seconds,frames)

    R_temp = [item[0] for item in Spo2]
    DC_R_comp=np.mean(R_temp)
    AC_R_comp=np.std(R_temp)
    # print(DC_R_comp)
    # print(R_temp)
    # print(AC_R_comp)
    I_r=AC_R_comp/DC_R_comp

    B_temp = [item[1] for item in Spo2]
    DC_B_comp=np.mean(B_temp)
    AC_B_comp=np.std(B_temp)

    # print(I_r)
    I_b=AC_B_comp/DC_B_comp
    SpO2_value=(A-B*((I_b*650)/(I_r*950)))
    return SpO2_value


# Grab images from the camera (separate thread)
def grab_images(cam_num, queue,self):
    global data
    global boxes
    global frameCount
    global globalCount
    global totalFrame
    global FaceDetectionFlag
    global Spo2Flag
    global bpm
    global hr
    global recordFlag
    global frontalFlag
    global frontFaceCount

    bpm=0
    hr=0
    # cap = cv2.VideoCapture(cam_num-1 + CAP_API)
    cap = cv2.VideoCapture(cam_num)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height) 

   
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
    name_final=''
    if EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    while capturing:
        if cap.grab():                             
            retval, image = cap.retrieve(0)
            fullScale = image.copy()
            recordVid = image.copy()
            fullScale = imutils.resize(fullScale,width=400,height=400)
            fullScale = cv2.cvtColor(fullScale,cv2.COLOR_BGR2RGB)
            if recordFlag:
                saveFrame = fullScale.copy()
            image = imutils.resize(image,width=400,height=400)
            boxFrame=image.copy()
            # print(queue.qsize())
            cv2.rectangle(boxFrame,(150,100),(250,200),(0,0,255),3)
            faceFrame = image[100:200,150:250]
            if frontalFlag and detectionFrontFace(faceFrame.copy()):
                print("")
                frontFaceCount=frontFaceCount+1
                if frontFaceCount==5:    
                    frontalFlag = False
                    frontFaceCount=0 
                    self.clickme()   
            # cv2.imshow("face",faceFrame)
            cv2.waitKey(1)
            if boxFrame is not None and (queue.qsize() < 2 or (Spo2Flag))  :
                # faceFrame = image[100:200,150:250]

                if frameCount==0 and (FaceDetectionFlag):
                    process=Process()

                    encodings = face_recognition.face_encodings(image, boxes)

                    for encoding in encodings:
                        matches = face_recognition.compare_faces(data["encodings"],
                        encoding)
                        name='Unknown'

                        if True in matches:
                            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
                            counts ={}

                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name]=counts.get(name,0)+1
                            print(counts)
                            name = max(counts,key=counts.get)

                        print(name)
                        self.label_5.setText("ID:"+name)
                        name_final=name
                        FaceDetectionFlag=0
                    if recordFlag:
                        today = date.today()
                        t = time.localtime()
                        current_time = time.strftime("%H%M%S", t)
                        print(current_time)
                        if os.path.exists("recordings/"+str(today)):
                            recording = cv2.VideoWriter("recordings/"+str(today)+'/'+str(name)+str(current_time)+'.avi',  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            20, (400,400))
                        else:
                            os.mkdir("recordings/"+str(today))
                            recording = cv2.VideoWriter("recordings/"+str(today)+str(current_time)+'/'+str(name)+str(current_time)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, (400,400))
                    thresh,mask=face_detect_and_thresh(faceFrame)

                    temp,min_value,max_value=spartialAverage(mask,faceFrame)
                    # print(temp)
                    # print(type(temp))
                    # print(str(type(temp)))
                    if(type(temp)==type(2)):
                        print("failed estimation, try again")
                        frameCount=totalFrame
                        Spo2Flag=2

                    final_sig.append(temp)

                elif (Spo2Flag==1) and frameCount<totalFrame and frameCount>1:
                    if recordFlag:
                        recordVid=cv2.resize(recordVid,(400,400))
                        recording.write(recordVid)
                    thresh,mask=face_detect_and_thresh(faceFrame)
                    process.frame_in = fullScale
                    process.run()
                    bpm=process.bpm
                    if process.bpms.__len__() > 50:
                        if(max(process.bpms-np.mean(process.bpms))<20 and bpm<100):
                            hr=np.mean(process.bpms)
                    if bpm>0:
                        print(bpm)
                        print(hr)
                    final_sig.append(MeanRGB(thresh,faceFrame,final_sig[-1],min_value,max_value))
                    self.label_1.setText("HeartRate:"+str(int(hr)))
                    
                    

                if frameCount==totalFrame:
                    if recordFlag:
                        recording.release()
                        recordFlag=False
                        
                        
                    if Spo2Flag==1:
                        result=SPooEsitmate(final_sig,totalFrame,totalFrame,duration) # the final signal list is sent to SPooEsitmate function with length of the video
                        print(result)
                        try:
                            self.label_2.setText("SPO2 Level:"+str(int(result)))
                        except:
                            self.label_2.setText("SPO2 Level:"+"NA")
                        tempFlag=checkPing(self.AI_CAN_IP)
                        
                        print(tempFlag)
                        if tempFlag==1:
                            sensorValue=get_value(self.AI_CAN_IP)
                            # print(sensorValue)
                            Ambient = stringGetValue(sensorValue,4) 
                            Compensated = stringGetValue(sensorValue,6) 
                            self.label_3.setText("Ambient:"+str((format(float(Ambient),'.2f'))))
                            self.label_4.setText("Compen.:"+str((format(float(Compensated),'.2f'))))
                            Ambient = format(float(Ambient),'.2f')
                            Compensated = format(float(Compensated),'.2f')
                        else:
                            Ambient = "NA"
                            Compensated = "NA"
                            self.label_3.setText("Ambient:"+Ambient)
                            self.label_4.setText("Compen.:"+Compensated)
                                
                        checkName(name_final,result,hr,Compensated,Ambient)
                        
                        # if url_ok():
                        #     upload()
                        Spo2Flag=0
                        # Webspo2Flag= not Webspo2Flag
                        

                    elif Spo2Flag==2:
                        
                        print("Try again with face properly aligned")

                # print(self.AI_CAM_IP)
                queue.put(boxFrame)
                frameCount=frameCount+1
                globalCount=globalCount +1 
                if globalCount%500==0:
                    tempFlag=checkPing(self.AI_CAN_IP)
                    frontalFlag = True
                    print(tempFlag)
                    if tempFlag==1:                       
                        sensorValue=get_value(self.AI_CAN_IP)
                        Ambient = stringGetValue(sensorValue,4) 
                        Compensated = stringGetValue(sensorValue,6) 
                        self.label_3.setText("Ambient:"+str((format(float(Ambient),'.2f'))))
                        self.label_4.setText("Compen.:"+str((format(float(Compensated),'.2f'))))
                        if((float(Compensated))>37.7):
                            send_mail()
                        
                    
                    else:
                        Ambient = "NA"
                        Compensated = "NA"
                        self.label_3.setText("Ambient:"+Ambient)
                        self.label_4.setText("Compen.:"+Compensated)
                    
                    if globalCount>100000:
                        globalCount=0
                    print(globalCount)
                    

                # print(frameCount)
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()

def check_url(url):
    if url == "0":
        camera=0
    elif url == "1":
        camera =1
    else:
        camera = "http://"+str(url)+"/mjpeg/1"
    return camera

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MyWindow(QMainWindow):

    text_update = pyqtSignal(str)
    # Create main window
    def __init__(self,IP,AI_CAN_IP,email,identifier, parent=None):
        
        QMainWindow.__init__(self, parent)
        
        self.central = QWidget(self)
        
        # Ipaddress,done1 = QInputDialog.getText( 
        #      self, 'Input Dialog', 'IP address:')
        self.AI_CAN_IP =AI_CAN_IP
        
        self.IP = IP
        self.textbox = QTextEdit(self.central)
        self.textbox.setFont(TEXT_FONT)
        self.textbox.setMinimumSize(300, 100)
        self.text_update.connect(self.append_text)
        sys.stdout = self
        print(identifier)
        print(IP)
        print(email)
        print("Camera number %u" % camera_num)
        print("Image size %u x %u" % IMG_SIZE)
        if DISP_SCALE > 1:
            print("Display scale %u:1" % DISP_SCALE)

        self.vlayout = QVBoxLayout()        # Window layout
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.label = QLabel(self)
        self.vlayout.addWidget(self.label)
        self.vlayout.addWidget(self.textbox)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.title = 'PyQt5 simple window - pythonspot.com'
        self.left = 640
        self.top = 360
        # self.left = 500
        # self.top = 500
        self.width = 640
        self.height = 480
        self.UiComponents()
        self.mainMenu = self.menuBar()      # Menu bar
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(exitAction)
        
       

    def UiComponents(self):

        global hr
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label_2 = QLabel('SPO2 Level:',self)
        self.label_2.move(500,50)
        self.label_2.setStyleSheet("border: 1px solid black;")
        
        self.label_1 = QLabel('heartRate:', self) 
        self.label_1.move(500, 75) 
        self.label_1.setStyleSheet("border: 1px solid black;")


        self.label_3 = QLabel('Ambient:',self)
        self.label_3.move(500,225)
        self.label_3.setStyleSheet("border: 1px solid black;")

        self.label_4 = QLabel('Compen.:',self)
        self.label_4.move(500,250)
        self.label_4.setStyleSheet("border: 1px solid black;")
         
        self.label_5 = QLabel('ID:',self)
        self.label_5.move(500,175)
        # self.label_5.resize(200,20) 
        self.label_5.setStyleSheet("border: 1px solid black;")
        
        # creating a push button
        button = QPushButton("SPO2", self)

        # setting geometry of button
        button.setGeometry(200, 150, 100, 30)
        button.move(500,125)
        # adding action to a button

        button.clicked.connect(self.clickme)

        button2 = QPushButton("Update", self)

        # setting geometry of button
        button2.setGeometry(200, 150, 100, 30)
        button2.move(500,300)
        # adding action to a button
        button2.clicked.connect(self.updateV)

        button3 = QPushButton("Record", self)

        # setting geometry of button
        button3.setGeometry(200, 150, 100, 30)
        button3.move(500,335)
        # adding action to a button
        button3.clicked.connect(self.record)

        self.show()

    

    def updateV(self):
        tempFlag=checkPing(self.AI_CAN_IP)
        print(tempFlag)
        if tempFlag==1:

            sensorValue=get_value(self.AI_CAN_IP)
            Ambient = stringGetValue(sensorValue,4) 
            Compensated = stringGetValue(sensorValue,6) 
            self.label_3.setText("Ambient:"+str((format(float(Ambient),'.2f'))))
            self.label_4.setText("Compen.:"+str((format(float(Compensated),'.2f'))))
        # if globalCount>100000:
            #     globalCount=0
            
            if(int(float(Compensated))>37):
                send_mail()
        
        else:
            Ambient = "NA"
            Compensated = "NA"
            self.label_3.setText("Ambient:"+Ambient)
            self.label_4.setText("Compen.:"+Compensated)

    def record(self):
        global recordFlag
        recordFlag = True
        self.clickme()


    def clickme(self):
        global hr,Spo2Flag,FaceDetectionFlag,frameCount,final_sig,spo2_set,name
        final_sig=[]
        name=[]
        spo2_set=[]
        frameCount=0
        Spo2Flag=1
        FaceDetectionFlag=1
        self.label_1.setText("HeartRate:" + str(hr))
        # printing pressed
        print("pressed")
    # Start image capture & display
    def start(self):
        IP = self.IP
        
        print(IP)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)        # Timer to trigger display
        self.timer.timeout.connect(lambda:
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)
        self.capture_thread = threading.Thread(target=grab_images,
                    args=(IP, image_queue,self))
        self.capture_thread.start()         # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size,
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1],
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Handle sys.stdout.write: update text display
    def write(self, text):
        self.text_update.emit(str(text))
    def flush(self):
        pass

    # Append to text display
    def append_text(self, text):
        cur = self.textbox.textCursor()     # Move cursor to end of text
        cur.movePosition(QTextCursor.End)
        s = str(text)
        while s:
            head,sep,s = s.partition("\n")  # Split line at LF
            cur.insertText(head)            # Insert text at cursor
            if sep:                         # New line if LF
                cur.insertBlock()
        self.textbox.setTextCursor(cur)     # Update visible cursor

    # Window is closing: stop video capture
    def closeEvent(self, event):
        global capturing
        capturing = False
        self.capture_thread.join()

class Window(QDialog): 
  
    # constructor 
    def __init__(self): 
        super(Window, self).__init__() 
  
        # setting window title 
        self.setWindowTitle("Setup") 
  
        # setting geometry to the window 
        self.setGeometry(100, 100, 300, 400) 
  
        # creating a group box 
        self.formGroupBox = QGroupBox("Enter user details") 
  
        # creating spin box to select age 
         
  
        # creating a line edit 
        self.nameLineEdit = QLineEdit() 

        # creating a line edit 
        self.emailLineEdit = QLineEdit() 

        # creating a line edit 
        self.iPLineEdit = QLineEdit()

        self.JsonIP = QLineEdit()
        
        # calling the method that create the form 
        self.createForm() 

        self.NoScanButton = QPushButton(self.tr("&No Scan"))


        # creating a dialog button for ok and cancel 
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel) 


        # adding action when form is accepted 
        self.buttonBox.accepted.connect(self.getInfo) 
  
        # addding action when form is rejected 
        self.buttonBox.rejected.connect(self.reject) 

        self.NoScanButton.clicked.connect(self.NoScan)

        # creating a vertical layout 
        mainLayout = QVBoxLayout() 
  
        # adding form group box to the layout 
        mainLayout.addWidget(self.formGroupBox) 
  
        # adding button box to the layout 
        mainLayout.addWidget(self.buttonBox) 

        mainLayout.addWidget(self.NoScanButton) 
  
        # setting lay out 
        self.setLayout(mainLayout) 
  

    def NoScan(self):
        print("Identifier : {0}".format(self.nameLineEdit.text())) 
        print("Email : {0}".format(self.emailLineEdit.text())) 
        print("IP : {0}".format(self.iPLineEdit.text())) 
        Identifier = self.nameLineEdit.text()
        Email = self.emailLineEdit.text()
        IP=check_url(self.iPLineEdit.text())
        AI_CAN_IP =  "http://"+self.JsonIP.text()
        print(AI_CAN_IP)
        userDetails = {"Identifier":Identifier,"Email":Email,"IP":IP,"AI_CAN_IP":AI_CAN_IP}
        with open('userData.pickle', 'wb') as f:
            pickle.dump(userDetails, f)
        # closing the window 
        self.close()
        win = MyWindow(IP,AI_CAN_IP,Email,Identifier)
        win.show()
        win.setWindowTitle(VERSION)
        win.start()



    # get info method called when form is accepted 
    def getInfo(self): 
  
        # printing the form information 
        print("Identifier : {0}".format(self.nameLineEdit.text())) 
        print("Email : {0}".format(self.emailLineEdit.text())) 
        print("IP : {0}".format(self.iPLineEdit.text())) 
        Identifier = self.nameLineEdit.text()
        Email = self.emailLineEdit.text()
        IP=self.iPLineEdit.text()
        AI_CAN_IP =  get_IP(Identifier)
        userDetails = {"Identifier":Identifier,"Email":Email,"IP":IP,"AI_CAN_IP":AI_CAN_IP}
        with open('userData.pickle', 'wb') as f:
            pickle.dump(userDetails, f)
        # closing the window 
        self.close()
        win = MyWindow(IP,AI_CAN_IP,Email,Identifier)
        win.show()
        win.setWindowTitle(VERSION)
        win.start()
        
         
  
    # creat form method 
    def createForm(self): 
  
        # creating a form layout 
        layout = QFormLayout() 
  
        # adding rows 
        # for name and adding input text 
        layout.addRow(QLabel("Identifier"), self.nameLineEdit) 
  
        # for degree and adding combo box 
        layout.addRow(QLabel("Email"), self.emailLineEdit) 
  
        # for age and adding spin box 
        layout.addRow(QLabel("Cam-IP"), self.iPLineEdit) 
  
        layout.addRow(QLabel("Json-IP"), self.JsonIP)

        # setting layout 
        self.formGroupBox.setLayout(layout) 

class SetupWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        layout = QGridLayout()
        self.setLayout(layout)

        self.radiobuttonSetup = QRadioButton("Setup")
        self.radiobuttonSetup.setChecked(True)
        self.radiobuttonSetup.state = "Setup"
        
        layout.addWidget(self.radiobuttonSetup, 0, 0)

        self.radiobuttonRun = QRadioButton("Run")
        self.radiobuttonRun.state = "Run"
        
        layout.addWidget(self.radiobuttonRun, 0, 1)

        button = QPushButton("Proceed", self) 
  
        # setting geometry of button 
        button.setGeometry(200, 150, 100, 30) 
        layout.addWidget(button, 1, 0)
        # adding action to a button 
        button.clicked.connect(self.onClicked) 
        
    def onClicked(self):
        global setupFlag
        if self.radiobuttonRun.isChecked():
            setupFlag=False
        else:
            setupFlag=True
        self.close()


if __name__ == '__main__':
    
    state = QApplication(sys.argv)
    screen = SetupWindow()
    screen.show()
    state.exec()
    
    if len(sys.argv) > 1:
        try:
            camera_num = int(sys.argv[1])
        except:
            camera_num = 0
    if camera_num < 1:
        print("Invalid camera number '%s'" % sys.argv[1])
    else:
        app = QApplication(sys.argv)
        if (path.exists("userData.pickle") and (not setupFlag)):
            with open('userData.pickle','rb') as f:

                userDetails = pickle.load(f)
                IP = userDetails.get('IP')
                
                Email = userDetails.get('Email')
                Identifier = userDetails.get("Identifier")
                AI_CAN_IP =  userDetails.get("AI_CAN_IP")
                win = MyWindow(IP,AI_CAN_IP,Email,Identifier)
                win.show()
                win.setWindowTitle(VERSION)
                win.start()
                sys.exit(app.exec())
        
        else:
            window = Window() 
            window.show()
            sys.exit(app.exec())
#EOF
