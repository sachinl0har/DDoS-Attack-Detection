import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
import customtkinter as ctk
#from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from PIL import Image, ImageTk

on = True

le = LabelEncoder()

root = ctk.CTk()
root.state('zoomed')
#root.title("DDOS ATTACK DETECTION SYSTEM")
#root.configure(background="white")

#app = customtkinter.CTk()
#root.geometry("500x750")
#root.title("DDoS Attack Detection System")



img1 = ImageTk.PhotoImage(Image.open("pattern.png"))
l1= ctk.CTkLabel(master=root,image = img1)
l1.pack()

frame2 = ctk.CTkFrame(root,width=900,height=400, border_color="white",border_width=1)
frame2.place(x=450,y=180)

frame5= ctk.CTkFrame(root,width=900,height=400, border_color="white",border_width=1)
frame5.place(x=450,y=180)


frame3 = ctk.CTkFrame(root,width=1000,height=100,border_color="white",border_width=1)
frame3.place(x=250,y=50)

frame4 = ctk.CTkFrame(root,width=900,height=400,border_color="white",border_width=1)
#frame4.place(x=980,y=150) 
frame4.place(x=450,y=180)

head = ctk.CTkLabel(frame3,text = "DDOS ATTACK DETECTION SYSTEM", corner_radius=6)
head.place(x=50,y=20)
"""
tpot_data = pd.read_csv(r'E:\IDS Dataset\KDD_CUP_2.csv')#, sep='COLUMN_SEPARATOR', dtype=np.float64)
tpot_data['protocol_type']=le.fit_transform(tpot_data['protocol_type'])
tpot_data['service']=le.fit_transform(tpot_data['service'])
tpot_data['flag']=le.fit_transform(tpot_data['flag'])

x = tpot_data.drop(['label'],axis=1)
y = tpot_data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
"""
new_data = pd.read_csv(r"Trainnew.csv")#, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
x_train = new_data.drop(["label"],axis=1)
y_train = new_data["label"]


new_data = pd.read_csv(r"Testnew.csv")#, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
x_test = new_data.drop(["label"],axis=1)
y_test = new_data["label"]



    
def SVM():
    
    
    
    model2 = SVC(kernel='linear',random_state=2)
    model2.fit(x_train, y_train)
    
    
    
    print("=" * 40)
    model2.fit(x_train, y_train)
    
    model2_pred = model2.predict(x_test)
    #print(model2_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model2_pred)))
    print("Accuracy : ",accuracy_score(y_test,model2_pred)*100)
    accuracy = accuracy_score(y_test, model2_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model2_pred) * 100)
    repo = (classification_report(y_test, model2_pred))
    
    label4 = ctk.CTkLabel(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",7))
    label4.place(x=205,y=100)
    
    #label5 = ctk.CTkLabel(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    #label5.place(x=205,y=320)
    from joblib import dump
    dump (model2,"SVM.joblib")
    print("Model saved as SVM.joblib")
    

    


    
    
def RF():
    
    #seed = 7
    frame4.tkraise();
    num_trees = 100
    max_features = 3
    
    model5 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

    
    
    model5.fit(x_train, y_train)
    
    model5_pred = model5.predict(x_test)

        
    print(model5_pred)
            
            
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, model5_pred)))
    print("Accuracy : ",accuracy_score(y_test,model5_pred)*100)
    accuracy = accuracy_score(y_test, model5_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, model5_pred) * 100)
    repo = (classification_report(y_test, model5_pred))
            
    classrepo= ctk.CTkLabel(frame4,text ="Classification report ",width=35,font=("Bahnschrift Light",27,"bold"))
    classrepo.place(x=50,y=20)
            
            
    label4 = ctk.CTkLabel(frame4,text =str(repo),font=("Tempus Sanc ITC",15))
    label4.place(x=85,y=85)
            
    label5 = ctk.CTkLabel(frame4,text ="Accuracy : "+str(ACC)+"%\nModel saved as RF.joblib",width=35,height=3,font=("Tempus Sanc ITC",16))
    label5.place(x=70,y=300)
    from joblib import dump
    dump (model5,"RF.joblib")
    print("Model saved as RF.joblib")  
        
    
    
    



    
def EXIT():
    root.destroy()



frame = ctk.CTkFrame(root,width=290,height=400,border_color="white",border_width=1)
frame.place(x=115,y=180)



#button2 = ctk.CTkButton(frame,command=SVM,text="SVM",bg="white",width=15,height=2,bd=3)
#button2.place(x=5,y=50)



button3 = ctk.CTkButton(frame,command= RF,text="TRAIN",width=210,corner_radius=6,fg_color="green")
button3.place(x=35,y=200)



TMState=tk.IntVar()
TMState=""

from tkinter import ttk
State_Name = {"SUPPORT VECTOR MACHINE":1,"RANDOM FOREST":2}
TMStateEL=ttk.Combobox(frame,values=list(State_Name.keys()),width=30)
TMStateEL.state(['readonly'])
TMStateEL.bind("<<ComboboxSelected>>", lambda event: print(State_Name[TMStateEL.get()]))

TMStateEL.current(0)
TMStateEL.place(x=45,y=120)

model_list = {"SUPPORT VECTOR MACHINE":"OLD_MODELS/SVM.joblib","RANDOM FOREST":"OLD_MODELS/RF.joblib"}


def ok():
    
    frame2.tkraise()
    print ("value is:" + TMStateEL.get())
    model_choice = TMStateEL.get()
    choosen_model = model_list[model_choice]
    print(choosen_model)
    from joblib import load
    ans = load(choosen_model)
    
    
    from tkinter.filedialog import askopenfilename
    fileName = askopenfilename(initialdir='E:/DDOS Attack 27 Feb 2023/DDOS Attack', title='Select DataFile For INTRUSION Testing',
                                       filetypes=[("all files", "*.csv*")])
    
    file =pd.read_csv(fileName)
    file['protocol_type']=le.fit_transform(file['protocol_type'])
    file['service']=le.fit_transform(file['service'])
    file['flag']=le.fit_transform(file['flag'])

    qn = file.drop(["label"],axis=1)
    
    A = ans.predict(qn)
    print(A)
    def listToString(s): 
    
        # initialize an empty string
        str1 = "" 
        
        # traverse in the string  
        for ele in s: 
            str1 += ele  
        
        # return string  
        return str1 
    print(listToString(A)) 
    B = listToString(A)
    
    if B == 'normal':
        frame2.tkraise()
        output = 'DDOS Attack Not Detected'
        frame2.configure(fg_color = 'green')
        
        attack = ctk.CTkLabel(frame2,text=str(output),width=30)
        attack.place(x=150,y=150)

    else:
        frame5.tkraise()
        output = 'DDOS Attack Detected'
        frame5.configure(fg_color = 'red')
        attack = ctk.CTkLabel(frame5,text=str(output),width=30)
        attack.place(x=150,y=150)
 

   

button4 = ctk.CTkButton(frame,command= ok,text="TEST",width=210,corner_radius=6,fg_color="green")
button4.place(x=35,y=150)

button5 = ctk.CTkButton(frame,command=EXIT,text="EXIT",width=210,corner_radius=6,fg_color="red")
button5.place(x=35,y=250)


root.mainloop()

