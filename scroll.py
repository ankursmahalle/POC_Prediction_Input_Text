import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import matplotlib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import *
import pandas as pd
from nltk import word_tokenize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import numpy as np
import regex as re

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


topmargin = 20
leftmargin = 20


class VerticalScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = ttk.Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        hscrollbar = ttk.Scrollbar(self, orient=HORIZONTAL)
        hscrollbar.pack(fill=X, side=BOTTOM, expand=FALSE)
        self.canvas = tk.Canvas(
            self,
            bd=0,
            highlightthickness=0,
            width=1500,
            height=500,
            yscrollcommand=vscrollbar.set,
            xscrollcommand=hscrollbar.set,
        )
        self.canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=self.canvas.yview)
        hscrollbar.config(command=self.canvas.xview)

        # Reset the view
        self.canvas.xview_moveto(600)
        self.canvas.yview_moveto(600)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = ttk.Frame(self.canvas, width=1000, height=1000)
        self.interior.bind("<Configure>", self._configure_interior)
        self.canvas.bind("<Configure>", self._configure_canvas)
        self.interior_id = self.canvas.create_window(
            0, 0, window=self.interior, anchor=NW
        )

    def _configure_interior(self, event):
        # Update the scrollbars to match the size of the inner frame.
        size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
        self.canvas.config(scrollregion=(0, 0, size[0], size[1]))
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            # Update the canvas's width to fit the inner frame.
            self.canvas.config(width=self.interior.winfo_reqwidth())

    def _configure_canvas(self, event):
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            # Update the inner frame's width to fill the canvas.
            self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())


class Window:
    def __init__(self, master, *args, **kwargs):
        self.master = master
        self.frame = VerticalScrolledFrame(master)
        self.frame.pack(expand=True, fill=tk.BOTH)
        self.label = ttk.Label(
            master, text="Shrink the window to activate the scrollbar."
        )
        self.label.pack()

        for i in range(25):
            self.label1 = ttk.Label(
                self.frame.interior,
            )
            self.label1.pack(padx=10, pady=5)

        self.enterText_label = tk.Label(
            self.frame.interior,
            text="Enter message with values:",
            fg="black",
            font=("Courier", 12),
        )
        self.enterText_label.place(x=15 + leftmargin, y=1 + topmargin)

        self.text_Box = tk.Text(self.frame.interior, width=50, height=4)
        self.text_Box.place(x=20 + leftmargin, y=30 + topmargin)

        self.var = tk.StringVar()

        # Label for Flashing the prediction results
        self.hint_label = tk.Label(
            self.frame.interior,
            textvariable=self.var,
            font=("Courier", 12, "bold"),
            fg="Orange",
        )
        self.hint_label.place(x=450 + leftmargin, y=80 + topmargin)

        self.message1 = tk.Message(
            self.frame.interior,
            text="Message format to predict the Diabetes:",
            font=("Courier", 8),
            fg="black",
            justify="left",
            width=400,
        )
        self.message1.place(x=15 + leftmargin, y=100 + topmargin)

        self.message2 = tk.Message(
            self.frame.interior,
            text="Sample Message: Predict diabetes if pregnancies are 6, glucose is 148, bp is 72, skin is 35, insulin is 0, bmi is 33, dpf is 1 and age is 50",
            font=("Courier", 8),
            fg="black",
            justify="left",
            width=400,
        )
        self.message2.place(x=15 + leftmargin, y=120 + topmargin)

        # Predict Diabetes Button
        self.button = tk.Button(
            self.frame.interior,
            text="Predict Diabetes",
            foreground="black",
            font=("Courier", 10, "bold"),
            command=self.submit,
        )
        self.button.place(x=450 + leftmargin, y=30 + topmargin)

        # Reset Button
        self.button2 = tk.Button(
            self.frame.interior,
            text="RESET",
            font=("Courier", 10, "bold"),
            command=self.reset,
        )
        self.button2.place(x=900, y=590)

    def submit(self):
        df = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
        df_rename = df.rename(
            columns={
                "Pregnancies": "Pregnancies",
                "Glucose": "Glucose",
                "BloodPressure": "Blood Pressure",
                "SkinThickness": "Skin Thickness",
                "Insulin": "Insulin",
                "BMI": "BMI",
                "DiabetesPedigreeFunction": "Pedigree Function",
                "Age": "Age",
            }
        )

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
        sent = self.text_Box.get("1.0", "end")
        if sent == "\n":
            self.hint_label["fg"] = "black"
            self.hint_label["font"] = ("Courier", 10)
            self.var.set("Enter some text")
            return
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
        features = [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Pedigree Function",
            "Age",
        ]

        X = df_rename[features]
        y = df_rename["Outcome"]
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X.values, y.values)
        # Sample input format for non-diabetic: 1 85 66 29 0 26.6 0.351 31
        # Sample input format for diabetic: 6 148 72 35 0 33.6 0.627 50
        # TODO :: Use the below input format to process and retrieve the values.
        # TODO :: Fill the values if not provided as an input.
        # TODO :: Display the graph in the tkinter UI
        # Sample input format for diabetic: Predict diabetes if pregnancies are 6, glucose is 148, blood pressu
        # Sample input format for diabetic: Predict diabetes if pregnancies are 6, glucose is 148, bp is 72, sk
        # Now, let's check that how well our outcome column is balanced
        oput = dtree.predict([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        print(oput)
        if oput == [0]:
            self.hint_label["fg"] = "green"
            self.var.set("You are NonDiabetic")
        else:
            self.hint_label["fg"] = "#FF3E96"
            self.var.set("You are Diabetic")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=7
        )
        fill_values = SimpleImputer(missing_values=0, strategy="mean")
        X_train = fill_values.fit_transform(X_train)
        X_test = fill_values.fit_transform(X_test)
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)
        f, ax = plt.subplots()
        plt.rc("axes", titlesize=8)
        rfc.feature_importances_
        (
            pd.Series(rfc.feature_importances_, index=X.columns).plot(
                kind="barh",
                title="Below graph shows all the available features and their importance in the dataset.",
                fontsize=7,
                color="#7D7D7D",
            )
        )
        global internal_canvas
        self.internal_canvas = FigureCanvasTkAgg(f, self.frame.interior)
        self.internal_canvas.draw()
        self.internal_canvas.get_tk_widget().place(x=20 + leftmargin, y=400 + topmargin)
        self.internal_canvas.get_tk_widget().pack()
        # toolbar = NavigationToolbar2Tk(canvas, win)
        # toolbar.update()
        self.internal_canvas._tkcanvas.place(
            x=20 + leftmargin, y=200 + topmargin, width=800, height=400
        )

    def reset(self):
        self.text_Box.delete("1.0", END)
        self.var.set("")
        self.internal_canvas.get_tk_widget().destroy()


root = tk.Tk()
root.title("AI POC")
window = Window(root)
root.mainloop()
