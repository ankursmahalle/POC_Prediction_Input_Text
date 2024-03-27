import matplotlib
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from nltk import word_tokenize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

win = Tk()

# getting screen width and height of display
width = win.winfo_screenwidth()
height = win.winfo_screenheight()

# setting frame size to full screen
win.geometry("%dx%d" % (width, height))
# win.geometry("1500x550")

# Main Page Label(Prediction)
label = tk.Label(text="Prediction", font=("Courier", 16, "bold"), fg="")
label.place(x=1, y=1)
label.pack()

# Enter Message Lable
label1 = tk.Label(
    win, text="Enter Message With Values:", fg="brown", font=("Courier", 12)
)
label1.place(x=5, y=1)


# Enter Message Textbox
text = tk.Text(win, width=50, height=4)
text.place(x=20, y=30)
# text.pack()

var = tk.StringVar()

# Label for Flashing the prediction results
label2 = tk.Label(win, textvariable=var, font=("Courier", 12, "bold"), fg="Orange")
label2.place(x=450, y=80)

message1 = tk.Message(
    win,
    text="Message format to predict the Diabetes:",
    font=("Courier", 8),
    fg="blue",
    justify="left",
    width=400,
    highlightbackground="yellow",
)
message1.place(x=20, y=100)

message2 = tk.Message(
    win,
    text="Sample Message: Predict diabetes if pregnancies are 6, glucose is 148, bp is 72, skin is 35, insulin is 0, bmi is 33, dpf is 1 and age is 50",
    font=("Courier", 8),
    fg="black",
    justify="left",
    width=400,
    background="#FFF8DC",
)
message2.place(x=20, y=120)


def submit():
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
    sent = text.get("1.0", "end")
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
    # tree.plot_tree(dtree, feature_names=features)
    # plt.show()
    # dtree.feature_importances_
    # (pd.Series(dtree.feature_importances_, index=X.columns).plot(kind="barh"))
    # plt.show()
    # oput = dtree.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    # oput = dtree.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
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
        label2["fg"] = "green"
        var.set("You are NonDiabetic")
    else:
        label2["fg"] = "#FF3E96"
        var.set("You are Diabetic")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=7
    )
    fill_values = SimpleImputer(missing_values=0, strategy="mean")
    X_train = fill_values.fit_transform(X_train)
    X_test = fill_values.fit_transform(X_test)
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)

    f, ax = plt.subplots()
    rfc.feature_importances_
    (
        pd.Series(rfc.feature_importances_, index=X.columns).plot(
            kind="barh",
            title="Below graph shows all the available features and their importance in the dataset.",
            fontsize=7,
            color="#F4A460",
        )
    )

    global canvas
    canvas = FigureCanvasTkAgg(f, win)
    canvas.draw()
    canvas.get_tk_widget().place(x=20, y=400)
    canvas.get_tk_widget().pack()
    # toolbar = NavigationToolbar2Tk(canvas, win)
    # toolbar.update()
    canvas._tkcanvas.place(x=20, y=180, width=800, height=400)


# Predict Diabetes Button
button = tk.Button(
    win,
    text="Predict Diabetes",
    foreground="black",
    font=("Courier", 10, "bold"),
    command=submit,
    bg="#F0E68C",
)
button.place(x=450, y=30)
# button.pack()


# Function to reset(clear) the data from GUI window.
def reset():
    text.delete("1.0", END)
    var.set("")
    canvas.get_tk_widget().destroy()


# Reset Button
button2 = tk.Button(
    win, text="RESET", font=("Courier", 10, "bold"), bg="#F0E68C", command=reset
)
button2.place(x=1200, y=600)

# title for GUI Frame
win.title("AI-ML POC")
win.mainloop()
