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

# + id="oLK7Y1jiNXDa"
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tempfile
import requests
import joblib
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Dense, GRU, Activation
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from livelossplot.tf_keras import PlotLossesCallback

np.random.seed(0)
plt.style.use("ggplot")

# + [markdown] id="HrzFsjRGGLzP"
# *Essential info about tagged entities*:
# - geo = Geographical Entity
# - org = Organization
# - per = Person
# - gpe = Geopolitical Entity
# - tim = Time indicator
# - art = Artifact
# - eve = Event
# - nat = Natural Phenomenon

# + colab={"base_uri": "https://localhost:8080/", "height": 622} id="mCKmz4SAbI_m" outputId="c27609c8-c89e-4736-b74d-6ff0d0a36f55"
url = "https://raw.githubusercontent.com/seanbenhur/NER/main/data/ner_dataset.csv"
data = pd.read_csv(url, encoding="latin1")
data = data.fillna(method="ffill")
data.head(20)

# + colab={"base_uri": "https://localhost:8080/"} id="riOztP-8NXHT" outputId="a050f4e8-742c-4387-c9e8-d860e613b787"
print("Unique words in corpus:", data["Word"].nunique())
print("Unique tags in corpus:", data["Tag"].nunique())

# + id="GjOGZ0hDGLzR" colab={"base_uri": "https://localhost:8080/"} outputId="3a9a3baf-2f34-44a9-b5cb-33dfd1bfc595"
words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)
word2idx_path = "/content/drive/MyDrive/data/NER/word2idx.joblib"
tag2idx_path = "/content/drive/MyDrive/data/NER/tag2idx.joblib"
batch_size = 64
epochs = 3
# enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

# + id="-bzWR6H5GLzS"
tags = list(set(data["Tag"].values))
num_tags = len(tags)


# + id="VdJst_g5NYY_"
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


# + id="nMUQLppspkPj"
getter = SentenceGetter(data)
sentences = getter.sentences

# + id="SvENHO18pkaQ"
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# + colab={"base_uri": "https://localhost:8080/"} id="fdzhK-v6GLzV" outputId="6d24ceda-c891-499a-a91a-21c154876f84"
joblib.dump("word2idx", word2idx_path)
joblib.dump("tag2idx", tag2idx_path)

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="R44g5T7NYp_H" outputId="c23ad444-2669-4037-879d-5b01483a4ac6"
plt.hist([len(s) for s in sentences], bins=50)
plt.show()

# + id="FS4u3CRkpkc1"
max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(GRU(units=100, return_sequences=True, recurrent_dropout=0.1))(
    model
)
model = TimeDistributed(Dense(num_tags))(model)
out = Activation("softmax", dtype="float32", name="predictions")(model)
model = Model(input_word, out)
model.summary()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
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

callbacks = [PlotLossesCallback(), early_stopping]

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

# + colab={"base_uri": "https://localhost:8080/"} id="6euqX7UHplG7" outputId="f3fe463f-b579-408f-d979-8194bc0bdd94"
model.evaluate(x_test, y_test)

# + colab={"base_uri": "https://localhost:8080/"} id="iwe07iACYqnf" outputId="861960ce-24e4-40d4-ad97-d10063770d35"
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print("export_path = {}\n".format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)

print("\nSaved model:")
# !ls -l {export_path}

# + colab={"base_uri": "https://localhost:8080/"} id="n7cWBOwrY979" outputId="4e287186-8681-4bd6-c920-10067988ba71"
# !saved_model_cli show --dir {export_path} --all

# + colab={"base_uri": "https://localhost:8080/"} id="spByDiOIZWgV" outputId="929cabe0-2822-4fb4-da43-a6c01e01c429"
# !echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | {SUDO_IF_NEEDED} tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | {SUDO_IF_NEEDED} apt-key add -
#!{SUDO_IF_NEEDED} apt update

# + colab={"base_uri": "https://localhost:8080/"} id="aXTMnrMyZeKc" outputId="8a3a3874-1642-4f94-835a-b07c3f649f01"
# !apt-get install tensorflow-model-server

# + id="nwq-9WT0ZmoE"
os.environ["MODEL_DIR"] = MODEL_DIR

# + id="rGP9_KqAZqK1"
# !nohup tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=NER_model \
#   --model_base_path="${MODEL_DIR}" >server.log 2>&1

# + colab={"base_uri": "https://localhost:8080/"} id="hN3iuOUwZvMs" outputId="b146a39f-1481-4d73-a1c7-736204940585"
# !tail server.log

# + colab={"base_uri": "https://localhost:8080/"} id="Tyg4mKOVplJ-" outputId="fe4970c3-dcbd-433d-d638-875431a6fa04"
i = np.random.randint(0, x_test.shape[0])  # 659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" * 30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w - 1], tags[true], tags[pred]))

# + id="GfJtESU7apIR"
