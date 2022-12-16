import matplotlib.pyplot as plt
from pandas import read_csv
from os import listdir
from numpy import isnan
from pickle import dump, load


def main():
    data = {}

    for file in listdir("data/raw"):
        print(f"Loading {file}")
        df = read_csv(f"data/raw/{file}")

        for index, race in df.iterrows():
            if race["Pos"] == "1" and not isnan(race["Wgt_frac"]):
                f = float(race["Wgt_frac"])

                if f not in data:
                    data[f] = 1
                else:
                    data[f] += 1

    with open("data/graph/weight fraction.pkl", "wb") as file:
        dump(data, file)


def show():
    with open("data/graph/weight fraction.pkl", "rb") as file:
        data = load(file)
        print("Data Loaded")

    X = [i*(3/1000)+0.7 for i in range(0, 101)]
    Y = [0 for _ in range(0, 101)]

    for wgtf, num in data.items():
        Y[int((wgtf-0.7)/(3/1000))] += num


    print(X, Y)
    print("Plotting...")
    plt.plot(X, Y)
    plt.show()

if __name__ == '__main__':
    # main()
    show()
