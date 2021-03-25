# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback

np.random.seed(0)
plt.style.use("ggplot")

# -

# *Essential info about tagged entities*:
# - geo = Geographical Entity
# - org = Organization
# - per = Person
# - gpe = Geopolitical Entity
# - tim = Time indicator
# - art = Artifact
# - eve = Event
# - nat = Natural Phenomenon


data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")


words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)

tags = list(set(data["Tag"].values))
num_tags = len(tags)


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [
            (w, p, t)
            for w, p, t in zip(
                s["Word"].values.tolist(),
                s["POS"].values.tolist(),
                s["Tag"].values.tolist(),
            )
        ]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
sentences = getter.sentences


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

# + colab={} colab_type="code" id="q7VfnnkXpkfS"
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# + colab={"base_uri": "https://localhost:8080/", "height": 330} colab_type="code" id="Aee3mCZ3pkkv" outputId="b7fb911b-21d1-43e6-adc9-bb2d8bdfb921"
input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(
    model
)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

chkpt = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=1,
    verbose=0,
    mode="max",
    baseline=None,
    restore_best_weights=False,
)

callbacks = [PlotLossesCallback(), chkpt, early_stopping]

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=3,
    callbacks=callbacks,
    verbose=1,
)


model.evaluate(x_test, y_test)


i = np.random.randint(0, x_test.shape[0])  # 659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" * 30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w - 1], tags[true], tags[pred]))
