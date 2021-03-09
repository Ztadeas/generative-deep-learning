import keras 
import numpy as np
from keras import layers
from keras import models
import pandas as pd 
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import optimizers
from keras import models

maxlen = 12
steps = 2


dir_path = "C:\\Users\\Tadeas\\Downloads\\generatingpoetry\\Gutenberg-Poetry.csv"

everyhthing = pd.read_csv(dir_path)

text = ""
_text = []
next_word = []
len_text = []
len_nextword = []
binary_nextwords= []

for i in range(75000):
  k = everyhthing["s"][i]
  text += k
  text += " "

m = text.split()

for i in range(0, len(m) - maxlen, steps):
  _text.append(" ".join(m[i: i+maxlen]))
  next_word.append(m[i+maxlen])

for x in m:
  if x not in len_text:
    len_text.append(x)

for q in next_word:
  if q not in len_nextword:
    len_nextword.append(q)


tokenizer = Tokenizer(num_words=len(len_text))
tokenizer.fit_on_texts(m)
seq = tokenizer.texts_to_sequences(_text)
data = pad_sequences(seq, maxlen=12)


for n in next_word:
  binary_nextwords.append(len_nextword.index(n))
  

y = to_categorical(binary_nextwords, num_classes=len(len_nextword))

print(y.shape)

data = np.reshape(data, (279778, 1, 12))

print(data.shape)

m = models.Sequential()
m.add(layers.LSTM(128, input_shape=(1, 12)))
m.add(layers.Dense(len(len_nextword), activation="softmax"))

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["acc"])

def testdata():
  a = 75005

  text = ""

  k = []

  while len(text.split()) < 12:
    text += everyhthing["s"][i]
    text += " "
    a += 1

  k.append(text)

  return k
 

def sample(preds, temp = 1.0):
  preds = np.asarray(preds).astype("float64")
  preds = np.log(preds) / temp
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)


for i in range(0, 100):
  m.fit(data, y, batch_size=64, epochs=1)
  test_text = testdata()
  print(" ".join(test_text))
  fin_text = test_text
  test_seq = tokenizer.texts_to_sequences(fin_text)
  test_data = pad_sequences(test_seq, maxlen=12)
  test_data = np.reshape(test_data, (1, 1, 12))
  pred = m.predict(test_data, verbose=0)[0]
  pred = sample(pred)
  print(pred)
  next_word = len_nextword[pred]
  test_text += " "
  test_text += next_word
  test_text = test_text[1:]



  



  

