import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
import customtkinter as ctk
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import seaborn as sns

conn = sqlite3.connect('evaluation.db')
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS TrainLogs (
                id INTEGER PRIMARY KEY,
                model TEXT,
                accuracy FLOAT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS TestLogs (
                id INTEGER PRIMARY KEY,
                model TEXT,
                result TEXT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS FileLogs (
                id INTEGER PRIMARY KEY,
                protocol_type TEXT,
                flag TEXT,
                service TEXT,
                is_ddos INTEGER,
                timestamp TEXT
                )''')


def log_train(model_name, accuracy):
    # Log train operation
    cursor.execute("INSERT INTO TrainLogs (model, accuracy) VALUES (?, ?)", (model_name, accuracy))
    conn.commit()

def log_test(model_name, result):
    # Log test operation
    cursor.execute("INSERT INTO TestLogs (model, result) VALUES (?, ?)", (model_name, result))
    conn.commit()

def log_file_data(protocol_type, flag, service, is_ddos):
    # Log file data along with DDoS information and timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO FileLogs (protocol_type, flag, service, is_ddos, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (protocol_type, flag, service, is_ddos, timestamp))
    conn.commit()

def show_analytics(model_name, report, accuracy, confusion_matrix):
    # frame4.tkraise()
    
    # classrepo= ctk.CTkLabel(frame4,text ="Classification report ",width=35,font=("Bahnschrift Light",27,"bold"))
    # classrepo.place(x=50,y=20)
            
    # label4 = ctk.CTkLabel(frame4,text = report, font=("Tempus Sanc ITC",15))
    # label4.place(x=85,y=85)
            
    # label5 = ctk.CTkLabel(frame4,text ="Accuracy : {:.2f}%\nModel saved as {}.joblib".format(accuracy, model_name), width=35, height=3, font=("Tempus Sanc ITC",16))
    # label5.place(x=70,y=300)

    # Plotting accuracy bar chart
    # fig1 = plt.figure(figsize=(8, 6))
    # plt.bar(["Accuracy"], [accuracy], color='green')
    # plt.title('Accuracy')
    # plt.ylabel('Percentage')
    # plt.ylim(0, 100)
    # plt.tight_layout()
    # canvas1 = FigureCanvasTkAgg(fig1, master=frame4)
    # canvas1.draw()
    # canvas1.get_tk_widget().place(x=100, y=350)

    # Plotting confusion matrix
    fig2 = plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame4)
    canvas2.draw()
    canvas2.get_tk_widget().place(x=550, y=350)

    # Show plots
    plt.show()

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

    for widget in frame4.winfo_children():
        widget.destroy()

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
    
    label5 = ctk.CTkLabel(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    from joblib import dump
    dump (model2,"SVM.joblib")
    print("Model saved as SVM.joblib")
    
    log_train("SVM", accuracy_score(y_test, model2_pred))
    log_test("SVM", classification_report(y_test, model2_pred))
    
    model2_pred = model2.predict(x_test) 
    cm = confusion_matrix(y_test, model2_pred)
    show_analytics("SVM", classification_report(y_test, model2_pred), accuracy_score(y_test, model2_pred), cm)


    
    
def RF():

    for widget in frame4.winfo_children():
        widget.destroy()
    
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

    log_train("Random Forest", accuracy_score(y_test, model5_pred))
    log_test("Random Forest", classification_report(y_test, model5_pred))

    model5_pred = model5.predict(x_test) 
    cm = confusion_matrix(y_test, model5_pred)
    show_analytics("Random Forest", classification_report(y_test, model5_pred), accuracy_score(y_test, model5_pred), cm)

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

model_list = {"SUPPORT VECTOR MACHINE":r"OLD_MODELS/SVM.joblib","RANDOM FOREST":r"OLD_MODELS/RF.joblib"}


def ok():
    for widget in frame4.winfo_children():
        widget.destroy()

    for widget in frame2.winfo_children():
        widget.destroy()

    for widget in frame5.winfo_children():
        widget.destroy()
    print("Choosing file...")
    frame2.tkraise()
    print("Value is:", TMStateEL.get())
    model_choice = TMStateEL.get()
    chosen_model = model_list[model_choice]
    print("Chosen model:", chosen_model)
    
    from joblib import load
    ans = load(chosen_model)
    
    # Print feature names used during training
    # print("Feature names during training:", ans.feature_names_in_)
    
    from tkinter.filedialog import askopenfilename
    fileName = askopenfilename(initialdir=r'D:/MCA/SEM 4/PROJECT/DDoS-Attack-Detection', title='Select DataFile For INTRUSION Testing',
                                       filetypes=[("all files", "*.csv*")])
    print("Selected file:", fileName)
    
    if not fileName:
        print("No file selected. Aborting.")
        return
    
    file = pd.read_csv(fileName)
    file['protocol_type'] = le.fit_transform(file['protocol_type'])
    file['service'] = le.fit_transform(file['service'])
    file['flag'] = le.fit_transform(file['flag'])

    qn = file.drop(["label"], axis=1)
    
    # Print feature names in the prediction dataset
    print("Feature names in prediction dataset:", list(qn.columns))
    
    A = ans.predict(qn)
    print("Prediction:", A)
    
    B = A[0]  # Assuming A is an array with a single prediction
    
    if B == 'normal':
        frame2.tkraise()
        output = 'DDOS Attack Not Detected'
        frame2.configure(fg_color='green')
        attack = ctk.CTkLabel(frame2, text=str(output), width=30)
        attack.place(x=150, y=150)
        is_ddos_value = 0
    else:
        frame5.tkraise()
        output = 'DDOS Attack Detected'
        frame5.configure(fg_color='red')
        attack = ctk.CTkLabel(frame5, text=str(output), width=30)
        attack.place(x=150, y=150)
        is_ddos_value = 1

    log_file_data(file['protocol_type'].iloc[0], file['flag'].iloc[0], file['service'].iloc[0], is_ddos_value)



def show_reports():
    for widget in frame4.winfo_children():
        widget.destroy()

    # Retrieve data from SQLite tables
    cursor.execute("SELECT * FROM TrainLogs")
    train_logs = cursor.fetchall()
    cursor.execute("SELECT * FROM TestLogs")
    test_logs = cursor.fetchall()
    cursor.execute("SELECT * FROM FileLogs")
    file_logs = cursor.fetchall()

    # Add heading for TrainLogs
    train_heading = ctk.CTkLabel(frame4, text="Train Logs", font=("Arial", 14, "bold"))
    train_heading.pack()

    # Display TrainLogs in a Treeview
    train_treeview = ttk.Treeview(frame4)
    train_treeview['columns'] = ('Model', 'Accuracy')
    train_treeview.heading('#0', text='ID')
    train_treeview.heading('Model', text='Model')
    train_treeview.heading('Accuracy', text='Accuracy')

    for log in train_logs:
        train_treeview.insert('', 'end', text=log[0], values=(log[1], log[2]))

    train_treeview.pack()

    # Add heading for TestLogs
    test_heading = ctk.CTkLabel(frame4, text="Test Logs", font=("Arial", 14, "bold"))
    test_heading.pack()

    # Display TestLogs in a Treeview
    test_treeview = ttk.Treeview(frame4)
    test_treeview['columns'] = ('Model', 'Result')
    test_treeview.heading('#0', text='ID')
    test_treeview.heading('Model', text='Model')
    test_treeview.heading('Result', text='Result')

    for log in test_logs:
        test_treeview.insert('', 'end', text=log[0], values=(log[1], log[2]))

    test_treeview.pack()

    # Add heading for FileLogs
    file_heading = ctk.CTkLabel(frame4, text="File Logs", font=("Arial", 14, "bold"))
    file_heading.pack()

    # Display FileLogs in a Treeview
    file_treeview = ttk.Treeview(frame4)
    file_treeview['columns'] = ('Protocol Type', 'Flag', 'Service', 'Is DDoS', 'Timestamp')
    file_treeview.heading('#0', text='ID')
    file_treeview.heading('Protocol Type', text='Protocol Type')
    file_treeview.heading('Flag', text='Flag')
    file_treeview.heading('Service', text='Service')
    file_treeview.heading('Is DDoS', text='Is DDoS')
    file_treeview.heading('Timestamp', text='Timestamp')

    for log in file_logs:
        file_treeview.insert('', 'end', text=log[0], values=(log[1], log[2], log[3], log[4], log[5]))

    file_treeview.pack()

    # Perform graphical analysis for SVM and RF models
    # Assuming you have stored the accuracy and model names in a dictionary
    model_accuracies = {"SVM": 90.5, "RF": 85.3}

    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

button4 = ctk.CTkButton(frame,command= ok,text="TEST",width=210,corner_radius=6,fg_color="green")
button4.place(x=35,y=150)

button6 = ctk.CTkButton(frame, command=show_reports, text="Reports", width=210, corner_radius=6, fg_color="blue")
button6.place(x=35, y=250)

button5 = ctk.CTkButton(frame,command=EXIT,text="EXIT",width=210,corner_radius=6,fg_color="red")
button5.place(x=35,y=300)


root.mainloop()

