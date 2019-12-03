import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == "__main__":

    with open('data_train.json') as data_train:
        data = json.load(data_train)

    onestarinfo = [0, 0, 0, 0]
    twostarinfo = [0, 0, 0, 0]
    threestarinfo = [0, 0, 0, 0]
    fourstarinfo = [0, 0, 0, 0]
    fivestarinfo = [0, 0, 0, 0]

    for x in data:
        if x["stars"] ==  1.0:
            if x["useful"] == 0:
                onestarinfo[0] += 1
            elif x["useful"] <= 2:
                onestarinfo[1] += 1
            elif x["useful"] <= 10:
                onestarinfo[2] += 1
            elif x["useful"] > 10:
                onestarinfo[3] += 1


        if x["stars"] ==  2.0:
            if x["useful"] == 0:
                twostarinfo[0] += 1
            if x["useful"] == 1:
                twostarinfo[1] += 1
            if x["useful"] == 2:
                twostarinfo[2] += 1
            if x["useful"] > 2:
                twostarinfo[3] += 1


        if x["stars"] ==  3.0:
            if x["useful"] == 0:
                threestarinfo[0] += 1
            if x["useful"] == 1:
                threestarinfo[1] += 1
            if x["useful"] == 2:
                threestarinfo[2] += 1
            if x["useful"] > 2:
                threestarinfo[3] += 1

        if x["stars"] ==  4.0:
            if x["useful"] == 0:
                fourstarinfo[0] += 1
            if x["useful"] == 1:
                fourstarinfo[1] += 1
            if x["useful"] == 2:
                fourstarinfo[2] += 1
            if x["useful"] > 2:
                fourstarinfo[3] += 1

        if x["stars"] ==  5.0:
            if x["useful"] == 0:
                fivestarinfo[0] += 1
            if x["useful"] == 1:
                fivestarinfo[1] += 1
            if x["useful"] == 2:
                fivestarinfo[2] += 1
            if x["useful"] > 2:
                fivestarinfo[3] += 1


    names = ['1star', '2star', '3star', '4star', '5star']
    values = [onestarinfo[0], twostarinfo[0], threestarinfo[0], fourstarinfo[0], fivestarinfo[0]]

    plt.figure(figsize=(9, 3))

    plt.bar(names, values)
    plt.show()




