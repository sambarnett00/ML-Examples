from tensorflow.keras.models import load_model
from pandas import read_csv
from os import listdir
from math import exp


factors = ["horse_runs", "Ovr_Btn_mean", "Wgt_frac", "jockey_win%", "trainer_win%", "cumul_prize_frac", "Form_mean", "Dec"]


def get_weights(model):
    hm = load_model(f"models/{model}")
    wall = hm.get_weights()
    return wall[0].reshape((wall[0].shape[0], )), wall[1][0]


def softmax(z):
    z = [x/sum(z) for x in z]
    esum = sum([exp(x) for x in z])
    return [exp(x)/esum for x in z]


def main():
    weights, bias = get_weights("Model #Omega")
    wealth = 1000

    for file in listdir("data/raw"):
        print(f"Loading {file}")
        df = read_csv(f"data/raw/{file}")

        raceName = None
        probs = []

        exclude = None
        for index, race in df.iterrows():
            name = race["Name"]

            if raceName is None or raceName != name:
                if exclude is None:
                    p = softmax([x[0] for x in probs])
                    horses = [[a -1/(d-1), d, pos] for a, (_, d, pos) in zip(p, probs)]
                    advs = [k*(dec-1) > 1 for k, dec, pos in horses]
                    print(advs.count(True))
                    if advs.count(True) <= 2:
                        w = wealth
                        for k, dec, pos in horses:
                            if k * (dec-1) > 1:
                                if pos == "1":
                                    wealth += w*k*(dec-1)/3
                                else:
                                    wealth -= w*k*(dec-1)/3

                        print(f"Race: {raceName}  |  Wealth: {wealth}")

                raceName = name
                probs = []
                exclude = None

            if name != exclude:
                if race[factors].hasnans:
                    exclude = name

                else:
                    c = sum([float(f) * w for f, w in zip(race[factors], weights)]) + bias
                    probs.append([c, float(race["Dec"]), race["Pos"]])


def test():
    for file in listdir("data/raw"):
        df = read_csv(f"data/raw/{file}")

        print(df[factors])


if __name__ == '__main__':
    main()
    # test()