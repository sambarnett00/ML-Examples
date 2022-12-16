from tensorflow.keras.models import load_model
from pandas import read_csv
from os import listdir
from math import inf
from numpy import isnan


def get_weights(model):
    hm = load_model(f"models/{model}")
    wall = hm.get_weights()
    return wall[0].reshape((wall[0].shape[0], )), wall[1][0]


def main():
    total = 0
    wins = 0
    weights, bias = get_weights("Model #Omega")
    factors = ["horse_runs", "Ovr_Btn_mean", "Wgt_frac", "jockey_win%", "trainer_win%", "cumul_prize_frac", "Form_mean", "Dec"]

    for file in listdir("data/raw"):
        print(f"Loading {file}")
        df = read_csv(f"data/raw/{file}")

        name = None
        pos = 0
        xscore = -inf
        exclude = False
        for index, race in df.iterrows():
            if race["Name"] != name:
                if name is not None and not exclude: ## if race is not excluded
                    total += 1

                    if pos == "1":
                        wins += 1

                pos = inf
                xscore = -inf
                name = race["Name"]
                exclude = False

            if race["Name"] == name and not exclude:  ## if race is not excluded
                score = 0
                for w, f in zip(weights, factors):
                    if isnan(race[f]): ## invalid
                        exclude = True
                        break

                    else:
                        score += w * float(race[f])

                else:
                    score += bias
                    if score > xscore:
                        xscore = score
                        pos = race["Pos"]

    print(f"TOTAL RACES: {total}\nNUMBER OF WINS: {wins}\nWIN %: {100*wins/total}")


if __name__ == '__main__':
    main()
