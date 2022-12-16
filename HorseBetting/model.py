from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax
from numpy import argmax
import pandas as pd


columns = ["horse_runs",
           "Ovr_Btn_mean",
           "Wgt_frac",
           "jockey_win%",
           "trainer_win%",
           "cumul_prize_frac",
           "Form_mean",
           "Dec"]


class HorseModel(Model):
    def __init__(self):
        super(HorseModel, self).__init__()

        self.dense1 = Dense(1, input_shape=(8,), activation="linear")
        self.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def call(self, inputs):
        return self.dense1(inputs)


def load_training_data(file):
    df = pd.read_csv(file)
    return df[columns].to_numpy(), df[["RPR"]].to_numpy()


def train():
    X, Y = load_training_data("data/training/dataOmega.csv")
    hm = HorseModel()

    training = 220000

    hm.fit(X[:training], Y[:training], epochs=10)
    hm.save("models/Model #Omega")

    print(hm.get_weights())


def test():
    hm = load_model("models/Model #X")

    X, Y = load_training_data("data/training/dataX.csv")
    hm.evaluate(X, Y)

    X, Y = load_training_data("data/training/dataX.csv")
    for yhat, y in zip(hm.predict(X[0:32]), Y[0:32]):
        print(f"E:{abs(y-yhat)} -- {yhat} -> {y}")


def predict():
    from numpy import array

    hm = load_model("model/Model#Log1")

    names, X, Y, dec = load_all_data("data/training/dataLog1.csv")
    data = {}

    for n, x, y in zip(names, X, Y):
        if n not in data:
            data[n] = [hm(x)]
        else:
            data[n].append(hm(x))

    arr = []
    curName = None
    for name, scores in data.items():
        if curName is None or name != curName:
            arr.append([])
            curName = name

        arr[-1].append(scores)

    predictions = hm.predict(array(arr))
    df = pd.Dataframe(predictions)
    df.to_csv("data/predictions/dataLog1.csv")


if __name__ == '__main__':
    train()
    # test()
    # predict()


"""
[array([[ 0.3204699 ],
       [ 0.6000935 ],
       [ 9.803617  ],
       [-0.2584782 ],
       [-0.23347501],
       [-8.051896  ],
       [ 0.1935474 ],
       [-0.07926619]], dtype=float32), array([11.724708], dtype=float32)]

"""