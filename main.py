import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
import tkinter
from tkinter import *
from tkinter import scrolledtext
from matplotlib import pyplot as plt
import tkinter.messagebox as mb

raw_data = fetch_rcv1()

# maximum value of n is 804413 (full data)
# n is a number of records
n = 1000
data = raw_data.data[0:n, :]
target = raw_data.target[0:n, :]

data_df = pd.DataFrame.sparse.from_spmatrix(data)
target_df = pd.DataFrame.sparse.from_spmatrix(data=target, columns=raw_data['target_names'])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


def calc_score(neighborsNum):
    classifier = MLkNN(k=neighborsNum)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    return accuracy_score(y_test, prediction)


def btn_click(bottom, top):
    score = []
    if bottom >= top:
        mb.showerror(title="Error", message="Invalid interval of k!")
        return
    k_range = range(bottom, top + 1)
    for k in k_range:
        score.append(calc_score(k))
    plt.plot(k_range, score, label="Accuracy of learning")
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Nearest Neighbors')
    plt.legend()
    plt.title("KNN - Accuracy score")
    plt.show()
    return


def btn2_click(_feature, _target):
    x = []
    zeros = 0
    nonzeros = 0
    for i in range(0, n):
        if target[i, _target] == 1:
            if data[i, _feature] == 0:
                zeros += 1
            else:
                x.append(data[i, _feature])
                nonzeros += 1
    total = zeros + nonzeros
    plt.hist(x)
    plt.title("Histogram of feature " + str(_feature) + " with target " + str(_target) + "\nNonzero values only ("
              + str(nonzeros) + " from " + str(total) + ")")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()
    return


root = Tk()
root.title("Working with RCV1 dataset")
w = 646
h = 680
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
root.geometry("%dx%d+%d+%d" % (w, h, x, y))

varBottom = StringVar(root)
varBottom.set('3')
varTop = StringVar(root)
varTop.set('25')

label1 = Label(root, text="General information about dataset: ")
label2 = Label(root, text="Few records: ", justify="left")
label3 = Label(root, text="First records of label matrix: ")
label4 = Label(root, text="Calculate accuracy of learning with number of nearest neighbors from ")
label5 = Label(root, text=" to ")

spinBoxBottom = Spinbox(root, from_=2, to=200, width=5, textvariable=varBottom)
spinBoxTop = Spinbox(root, from_=2, to=200, width=5, textvariable=varTop)

btn = Button(root, text="Calculate", command=lambda: btn_click(int(spinBoxBottom.get()), int(spinBoxTop.get())))

txt1 = scrolledtext.ScrolledText(root, width=78, height=10)
txt2 = scrolledtext.ScrolledText(root, width=78, height=8)
txt3 = scrolledtext.ScrolledText(root, width=78, height=8)

varF = StringVar(root)
varF.set('0')
spinBoxF = Spinbox(root, from_=0, to=47235, width=5, textvariable=varF)

varT = StringVar(root)
varT.set('102')
spinBoxT = Spinbox(root, from_=0, to=102, width=5, textvariable=varT)

btn2 = Button(root, text="Show",
              command=lambda: btn2_click(int(spinBoxF.get()), int(spinBoxT.get())))

label7 = Label(root, text="Feature: ")
label8 = Label(root, text="Target: ")
label6 = Label(root, text="Histogram of data in feature, that have this target.")

label1.grid(column=0, row=0, columnspan=6, pady=10)
txt1.grid(column=0, row=1, columnspan=6)
label2.grid(column=0, row=2, columnspan=6, pady=10)
txt2.grid(column=0, row=3, columnspan=6)
label3.grid(column=0, row=4, columnspan=6, pady=10)
txt3.grid(column=0, row=5, columnspan=6)
label4.grid(column=0, columnspan=2, row=6, pady=10)
spinBoxBottom.grid(column=2, row=6, pady=10)
label5.grid(column=3, row=6, pady=10)
spinBoxTop.grid(column=4, row=6, pady=10)
btn.grid(column=5, row=6, pady=10)
label6.grid(column=0, row=7, pady=10)
label7.grid(column=1, row=7, pady=10)
label8.grid(column=3, row=7, pady=10)
spinBoxF.grid(column=2, row=7, pady=10)
spinBoxT.grid(column=4, row=7, pady=10)
btn2.grid(column=5, row=7, pady=10)


txt1.insert(INSERT, raw_data['DESCR'])
txt2.insert(INSERT, data_df.head())
txt3.insert(INSERT, target_df.head())

root.mainloop()

