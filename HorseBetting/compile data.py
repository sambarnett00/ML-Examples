import pandas as pd
import numpy as np
from os import listdir


if __name__ == "__main__":
    columns = ["horse_runs",
               "Ovr_Btn_mean",
               "Wgt_frac",
               "jockey_win%",
               "trainer_win%",
               "cumul_prize_frac",
               "Form_mean"]

    othercols = ["Dec", "RPR"]

    files = listdir("data/raw")
    saveDst = "dataOmega"
    arr = []

    for file in files:
        df = pd.read_csv(f"data/raw/{file}")
        print(f"Compiling data from {file}")
        allowIndex = [v != 0 for v in df["horse_runs"]]
        allow = {}

        for i, race in enumerate(df["Name"]):
            if not allowIndex[i]:
                allow[race] = -1

            else:
                if race not in allow:
                    allow[race] = 1

                elif allow[race] != -1:
                    allow[race] += 1


        p = 0
        for index, race in df.iterrows():
            n = allow[race["Name"]]
            if n > 1:
                if race["Pos"] in ["PU", "UR", "DSQ", "RR", "F", "REF", "BD", "SU", "LFT", "RO"]:
                    race["Pos"] = p+1
                    p += 1

                p = int(race["Pos"])
                rows = [race[c] for c in columns]
                if not np.any(np.isnan(rows)):
                    if not np.any([np.isnan(race[oc]) for oc in othercols]):
                        otherrows = [float(race["Dec"])-1, int(race["RPR"]) + 10*(race["Pos"]=="1")]

                        arr.append([race["Name"]] + rows + otherrows)


    df = pd.DataFrame(arr, columns=["Name"]+columns+othercols)
    df.to_csv(f"data/training/{saveDst}.csv", index=False)
