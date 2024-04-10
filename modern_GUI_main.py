import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
import customtkinter
#from tkvideo import tkvideo
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
#root = tk.Tk()
#root.configure(background="brown")
# root.geometry("1300x700")


#w, h = root.winfo_screenwidth(), root.winfo_screenheight()
#root.geometry("%dx%d+0+0" % (w, h))
#root.title("DDOS ATTACK DETECTION SYSTEM")

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")


root = customtkinter.CTk()
root.geometry("500x350")

# 43
#video_label =tk.Label(root)
#video_label.pack()
# read video to display on label
#player = tkvideo("studentvideo.mp4", video_label,loop = 1, size = (w, h))
#player.play()
# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
# image2 = Image.open('img2.jpg')
# image2 = image2.resize((1530, 900), Image.ANTIALIAS)

# background_image = ImageTk.PhotoImage(image2)

# background_label = tk.Label(root, image=background_image)

# background_label.image = background_image

# background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
#
#label_l1 = tk.Label(root, text="DDOS ATTACK DETECTION SYSTEM",font=("Times New Roman", 35, 'bold'),
                   # background="Maroon", fg="white", width=60, height=2)
#label_l1.place(x=0, y=0)

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady =60, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="DDoS Detection", font=("Roboto", 24, 'bold'))
label.pack(pady =12, padx = 10)


#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#img=ImageTk.PhotoImage(Image.open("dd11.jpg"))

#img2=ImageTk.PhotoImage(Image.open("dd2.jpg"))

#img3=ImageTk.PhotoImage(Image.open("dd3.jpg"))


#logo_label=tk.Label()
#logo_label.place(x=5,y=120)



# using recursion to slide to next image

#x = 1

# function to change to next image
#def move():
#	global x
#	if x == 4:
#		x = 1
#	if x == 1:
#		logo_label.config(image=img,width=2000,height=700)
#	elif x == 2:
#		logo_label.config(image=img2,width=2000,height=700)
#	elif x == 3:
#		logo_label.config(image=img3,width=2000,height=700)
#	x = x+1
#	root.after(2000, move)

# calling the function
#move()
#

def reg():
    from subprocess import call
    call(["python","modern_face_registration.py"])

def log():
    from subprocess import call
    call(["python","modern_face_login.py"])
    
def window():
  root.destroy()


button1 = customtkinter.CTkButton(master= frame, text="Login", command=log)
button1.pack(pady= 12, padx =10)

button2 = customtkinter.CTkButton(master= frame, text="Register",command=reg)
button2.pack(pady= 12, padx =10)

button3 = customtkinter.CTkButton(master= frame, text="Exit",command=window)
button3.pack(pady= 12, padx =10)

root.mainloop()