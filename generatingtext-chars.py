import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


dir_path = "C:\\Users\\Tadeas\\Downloads\\generatingpoetry\\Gutenberg-Poetry.csv"

everyhthing = pd.read_csv(dir_path)

text = ""

chars = ""

howmanychars = 100

main_text = []
target = []
maxlen = 50
data = []
y = []
len_truechars = []

for i in range(30000):
  text += everyhthing["s"][i]

for x in text:
  if x not in chars:
    chars += x

for w in range(0, len(text) - maxlen, 2):
  main_text.append(text[w: maxlen + w])
  target.append(text[w+maxlen])


for x in target:
  y.append(chars.index(x))
  
print("First part done")

tokenizer = Tokenizer(num_words= len(chars) + 1)
tokenizer.fit_on_texts(chars)

for i in main_text:
  p = i.split()
  send = []
  for x in range(len(p)):
    s = tokenizer.texts_to_sequences(p[x])
    if [] in s:
      pass
    
    else:
      send += s

  data.append(send) 
  send = []

main_data = pad_sequences(data, maxlen=50)

print(main_data.shape)

for h in target:
  if h not in len_truechars:
    len_truechars.append(h)


y = to_categorical(y, dtype="float32")

print(y.shape)

m = models.Sequential()
m.add(layers.LSTM(128, input_shape=(maxlen, 1)))
m.add(layers.Dense(len(chars), activation="softmax"))
m.compile(optimizer= optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["acc"])
m.fit(main_data, y, batch_size=64, epochs= 50)

def sample(preds, temp = 1.0):
  preds = np.asarray(preds).astype("float64")
  preds = np.log(preds) / temp
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def test_func():
  q = 30001
  test_tx = ""
  while len(test_tx) < 50:
    test_tx += everyhthing["s"][q]
    q += 1

  return test_tx

test_text = test_func()

for i in range(howmanychars):
  print(test_text)
  seq = tokenizer.texts_to_sequences(test_text)
  test_data = pad_sequences(seq, maxlen=50)
  pred = m.predict(test_data, verbose=0)[0]
  next_ind = sample(pred)
  next_char = chars[next_ind]
  test_text += next_char
  test_text = test_text[1:]
  

  



