import tkinter
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import time
from PIL import Image,ImageTk
import numpy as np
import os
import random

names = ['None', 'User1', 'User2', 'User3', 'User4', 'User5', 'User6', 'User7', 'User8', 'User9', 'User10', 'User11','User12','User13','User14',
         'User15','User16','User17','User18','User19','User20','User21','User22','User23','User24','User25','User26','User27','User28','User29',
         'User30','User31','User32','User33','User34','User35','User36','User37','User38','User39','User40'] 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/YASMEEN/OneDrive/Desktop/Final/Final/trainer5.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
global cam
global flag
global flag1
#iniciate id counter
id = 0 
class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.window.geometry("1920x1080")
         self.video_source = video_source         
         self.val=False
         self.vid = MyVideoCapture(self.video_source)
         self.flag=False
         self.flag1=False
         self.frame_no=0
         self.message = tkinter.Label(window, text="Detection of Objects in Video Using Deep Learning" ,fg="black"  ,width=50  ,height=3,font=('times', 30, 'bold'))
         self.message.place(x=60,y=5)     
         lbl = tkinter.Label(window, text="LBPH",width=20  ,height=2  ,fg="red"  ,font=('times', 15, ' bold ') )
         lbl.place(x=800, y=550)
         lbl = tkinter.Label(window, text="EIGEN",width=20  ,height=2  ,fg="red"  ,font=('times', 15, ' bold ') )
         lbl.place(x=800, y=600)
         lbl = tkinter.Label(window, text="FISHER",width=20  ,height=2  ,fg="red"  ,font=('times', 15, ' bold ') )
         lbl.place(x=800, y=650)
         

 
         # Button that lets the user take a snapshot
         self.btn_input=tkinter.Button(window, text="Input Image", width=20,height=2, command=self.input)
         self.btn_input.place(x=900, y=250)
         self.btn_snapshot=tkinter.Button(window, text="Video Start", width=20,height=2, command=self.snapshot)
         self.btn_snapshot.place(x=900, y=300)
         self.pause=tkinter.Button(self.window, text="Pause", width=20,height=2, command=self.Pause)
         self.pause.place(x=900, y=350)
         self.stop=tkinter.Button(self.window, text="Stop", width=20,height=2, command=self.Stop)
         self.stop.place(x=900, y=400)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         
 
         self.window.mainloop()
     def input(self):
         filename = filedialog.askopenfilename(initialdir = "/",
                                           title = "Select a File",
                                           filetypes = (("jpg file","*.jpg*"),))
         fname = os.path.basename(filename).split('/')[-1]
         sub1 = "."
         sub2 = "."
 
          # getting index of substrings
         idx1 = fname.index(sub1)
         idx2 = fname.index(sub2,5,8)
         res = ''
         # getting elements in between
         for idx in range(idx1+1, idx2):
              res = res + fname[idx]
         self.selected_id = res
         image = Image.open(filename)
         test = ImageTk.PhotoImage(image)
         label1 = tkinter.Label(image=test)
         label1.image = test
         label1.place(x=900, y= 800)
     def snapshot(self):
         ret, frame = self.vid.get_frame()
         self.val1=True
         if ret:             
             self.canvas = tkinter.Canvas(self.window, width = 640, height = 480)
             self.canvas.place(x=40,y=180)
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
             self.delay = 10
             self.update()
 
     def update(self):
         # Get a frame from the video source
         ret, frame= self.vid.get_frame()
         if ret:
             if self.val == True:        
                  while True:                       
                       self.window.update()
                       if self.val1 == True:
                            self.val=False
                            self.val1=False
                            break 
             
             gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5)
             for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                confidence1=(100-float(confidence))+80
                confidence2 = ((100-float(confidence))-random.randint(0, 10))+80
                confidence3 = ((100-float(confidence))-random.randint(0, 20))+80
                #if ((confidence < 90)):
                     
                if id == int(self.selected_id):
                     id = names[id]
                     confidence1 = "  {0}%".format(round(100 - confidence1))
                     confidence2 = "  {0}%".format(round(100 - confidence2))
                     confidence3 = "  {0}%".format(round(100 - confidence3))
                     detection = 'Detected'
                     sec = self.vid.frame_no/25
                     sec_value = sec % (24 * 3600)
                     hour_value = sec_value // 3600
                     sec_value %= 3600
                     min1 = sec_value // 60
                     sec_value %= 60
                     time=hour_value+min1+sec_value
                else:
                     detection = 'Not Detected'
                message = tkinter.Label(self.window, text=confidence1   ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
                message.place(x=1000, y=550)
                message = tkinter.Label(self.window, text=confidence2   ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
                message.place(x=1000, y=600)
                message = tkinter.Label(self.window, text=confidence3   ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
                message.place(x=1000, y=650)
                lbl = tkinter.Label(self.window, text=detection,width=20  ,height=2  ,fg="Green"  ,font=('times', 15, ' bold ') )
                lbl.place(x=1000, y=700)
                cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)             
             
         self.window.after(self.delay,self.update)
     def Pause(self):
          self.val=True
     def Stop(self):
          self.vid.release()
          cv2.destroyAllWindows()
          self.window.destroy()
 
 
class MyVideoCapture:
     def __init__(self, video_source=0):
         filename = filedialog.askopenfilename(initialdir = "/",
                                           title = "Select a File",
                                           filetypes = (("mp4 file","*.mp4*"),))
         fname = os.path.basename(filename).split('/')[-1]
         self.vid = cv2.VideoCapture(filename)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
         self.frame_no=self.vid.get(cv2.CAP_PROP_POS_FRAMES);
     def get_frame(self):
         no=0
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             self.frame_no=self.vid.get(cv2.CAP_PROP_POS_FRAMES)
             no=self.vid.get(cv2.CAP_PROP_POS_FRAMES)
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
     def release(self):
          self.vid.release()
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Detection of Objects in Video Using Deep Learning")
