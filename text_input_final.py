import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk import word_tokenize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure



class Window(tk.Tk):
    
    #df = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
    def __init__(self, master):
        self.master = master
        
        # Frame
        self.frame = tk.Frame(self.master, width=500, height=500)
        self.frame.pack()

        # Label
        self.label = tk.Label(self.frame, text="Enter Text")
        self.label.place(x=30, y=50)
        # Entry Text
        self.entry = tk.Entry(self.frame, width= 50 , font =('Arial',10))
        self.entry.place(x=10, y=80)
        # Button
        self.button = tk.Button(self.frame, text="Submit", command=self.submit)
        self.button.place(x=60, y=120)
        
        #GraphButton
        self.button1 = tk.Button(self.frame, text="Get Important Features Graph", command=PageThree)
        self.button1.place(x=90, y=160)

    def submit(self):
        df = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
        cols_list = df.columns.tolist()
        print(cols_list)
        # sent = "Show me the diabetic patients different age groups"

        pregnancies = 1
        glucose = 85
        bp = 66
        skin = 29
        insulin = 0
        bmi = 26.6
        dpf = 0.351
        age = 32

        sent = self.entry.get()
        myList = word_tokenize(sent)
        print("myList ==>>", myList)
        # TODO :: Add here the process 'myList' to get the values

        for i in range(len(myList)):
            item = myList[i]
            if item == "pregnancies":
                for j in range(len(myList)):
                    if j > i:
                        if myList[j].isdecimal():
                            pregnancies = myList[j]
                            break
            else:
                if item == "glucose":
                    for j in range(len(myList)):
                        if j > i:
                            if myList[j].isdecimal():
                                glucose = myList[j]
                                break
                else:
                    if item == "bp":
                        for j in range(len(myList)):
                            if j > i:
                                if myList[j].isdecimal():
                                    bp = myList[j]
                                    break
                    else:
                        if item == "skin":
                            for j in range(len(myList)):
                                if j > i:
                                    if myList[j].isdecimal():
                                        skin = myList[j]
                                        break
                        else:
                            if item == "insulin":
                                for j in range(len(myList)):
                                    if j > i:
                                        if myList[j].isdecimal():
                                            insulin = myList[j]
                                            break
                            else:
                                if item == "bmi":
                                    for j in range(len(myList)):
                                        if j > i:
                                            if myList[j].isdecimal():
                                                bmi = myList[j]
                                                break
                                else:
                                    if item == "dpf":
                                        for j in range(len(myList)):
                                            if j > i:
                                                if myList[j].isdecimal():
                                                    dpf = myList[j]
                                                    break
                                    else:
                                        if item == "age":
                                            for j in range(len(myList)):
                                                if j > i:
                                                    if myList[j].isdecimal():
                                                        age = myList[j]
                                                        break

        print("pregnancies = ", pregnancies)
        print("glucose = ", glucose)
        print("bp = ", bp)
        print("skin = ", skin)
        print("insulin = ", insulin)
        print("bmi = ", bmi)
        print("dpf = ", dpf)
        print("age = ", age)

        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']

        df_copy = df.copy(deep = True)
        df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
        # Showing the Count of NANs
        print(df_copy.isnull().sum())
        X = df[features]
        y = df['Outcome']

        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X.values, y.values)

        # tree.plot_tree(dtree, feature_names=features)
        # plt.show()
        dtree.feature_importances_
        (pd.Series(dtree.feature_importances_, index=X.columns).plot(kind='barh'))
        plt.show()

        # oput = dtree.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        # oput = dtree.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])

        # Sample input format for non-diabetic: 1 85 66 29 0 26.6 0.351 31
        # Sample input format for diabetic: 6 148 72 35 0 33.6 0.627 50

        # TODO :: Use the below input format to process and retrieve the values.
        # TODO :: Fill the values if not provided as an input.
        # TODO :: Display the graph in the tkinter UI
        # Sample input format for diabetic: Predict diabetes if pregnancies are 6, glucose is 148, blood pressure is 72, skin thickness is 35, insulin is 0, bmi is 33.6, diabetes pedigree function is 0.627 and age is 50
        # Sample input format for diabetic: Predict diabetes if pregnancies are 6, glucose is 148, bp is 72, skin is 35, insulin is 0, bmi is 33, dpf is 1 and age is 50
        
        # Now, let's check that how well our outcome column is balanced
        
        oput = dtree.predict([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        print(oput)
        self.label2 = tk.Label(self.frame, text = oput)
        self.label2.place(x=30, y=50)
        print("[1] means 'Diabetic'")
        print("[0] means 'Non Diabetic'")
        
        
        #p = df.hist(figsize = (20,20))


class PageThree(tk.Frame):
    def __init__(self, master):
        

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

root = tk.Tk()
root.title("AI POC")
window = Window(root)
root.mainloop()
