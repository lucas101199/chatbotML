import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle


def toFile(file, path):
    data = json.load(file)
    count = 1
    for sentences, intent in data:
        f = open(os.path.join(path, str(count) + ".txt"), 'w')
        f.write(sentences)
        f.close()
        count += 1

"""
def get_string_labels(predicted_scores_batch):
    predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
    predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
    return predicted_labels
"""

train_file = open("archive_gavagai/is_train.json", 'r')
train_file_oos = open("archive_gavagai/oos_train.json", 'r')

test_file = open("archive_gavagai/is_test.json", 'r')
test_file_oos = open("archive_gavagai/oos_test.json", 'r')

val_file = open("archive_gavagai/is_val.json", 'r')
val_file_oos = open("archive_gavagai/oos_val.json", 'r')

"""
data = pd.read_json(train_file)
data = data.append(pd.read_json(train_file_oos))
data = data.rename(columns={0: 'sentence', 1: 'scope'})
li = data['scope'].tolist()
l = []
for i in li:
    if i not in l:
        l.append(i)

data['scope'] = data['scope'].replace(l[:-1], 0)
data['scope'] = data['scope'].replace('oos', 1)


toFile(train_file, "archive_gavagai/train/in_scope/")
toFile(train_file_oos, "archive_gavagai/train/oos/")
toFile(test_file, "archive_gavagai/test/in_scope/")
toFile(test_file_oos, "archive_gavagai/test/oos/")
toFile(val_file, "archive_gavagai/val/in_scope/")
toFile(val_file_oos, "archive_gavagai/val/oos/")
"""
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'archive_gavagai/train',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'archive_gavagai/val',
    batch_size=batch_size,
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'archive_gavagai/test',
    batch_size=batch_size)

BUFFER_SIZE = 160000
BATCH_SIZE = 64

train_dataset = raw_train_ds.shuffle(BUFFER_SIZE)
test_dataset = raw_test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

bias = np.log([15000/100])
output_bias = tf.keras.initializers.Constant(bias)
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=raw_val_ds,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(raw_test_ds)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


X = []
for sent, label in json.load(test_file):
    X.append(sent)
for sent, label in json.load(test_file_oos):
    X.append(sent)

y_pred = model.predict(X)
print(y_pred)
metric = tfa.metrics.F1Score(num_classes=2)

print(metric)

"""
for sent, label in json.load(val_file_oos):
    predictions1 = model.predict(np.array([sent]))
    pred_lab = get_string_labels(predictions1)
    print(pred_lab.numpy())



data = json.load(f)
te = json.load(t)
data_oov = json.load(oov_train)
test_oov = json.load(tes)
val_oov = json.load(oov_val)
v = json.load(val)

intent = []
sentences = []

t_intent = []
t_sentences = []

e_intent = []
e_sentences = []

BUFFER_SIZE = 10000
BATCH_SIZE = 64
inttostring = {}
test_inttostring = {}

for sent, inte in te:
    t_intent.append(0)
    t_sentences.append(sent)
for sent, inte in test_oov:
    t_intent.append(1)
    t_sentences.append(sent)

# train data in scope and oov
for sent, inte in data:
    intent.append(0)
    sentences.append(sent)
for sent, inte in data_oov:
    intent.append(1)
    sentences.append(sent)

# eval data in scope and oov
for sent, inte in v:
    e_intent.append(0)
    e_sentences.append(sent)
for sent, inte in val_oov:
    e_intent.append(1)
    e_sentences.append(sent)


lbl_encoder = LabelEncoder()
lbl_encoder.fit(intent)
intent = lbl_encoder.transform(intent)

test_intent = lbl_encoder.transform(t_intent)
eval_intent = lbl_encoder.transform(e_intent)

vocab_size = 1000
embedding_dim = 64
max_len = 50
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequencest = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequencest, truncating='post', maxlen=max_len)

tokenizer.fit_on_texts(t_sentences)
word_index_test = tokenizer.word_index
sequencestest = tokenizer.texts_to_sequences(t_sentences)
padded_sequences_t = pad_sequences(sequencestest, truncating='post', maxlen=max_len)

tokenizer.fit_on_texts(e_sentences)
word_index_e = tokenizer.word_index
sequencestese = tokenizer.texts_to_sequences(e_sentences)
padded_sequences_e = pad_sequences(sequencestese, truncating='post', maxlen=max_len)

model = tf.keras.Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

model.summary()

epochs = 50
history = model.fit(padded_sequences, np.array(intent), validation_data=(padded_sequences_e, eval_intent), epochs=epochs)

loss, accuracy = model.evaluate(padded_sequences_t, np.array(test_intent))

print("Loss: ", loss)
print("Accuracy: ", accuracy)


dataset = tf.data.Dataset.from_tensor_slices((sentences, intent))
testset = tf.data.Dataset.from_tensor_slices((t_sentences, t_intent))


VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(dataset.map(lambda text, label: text))
print(dataset)

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(inttostring))
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(dataset, epochs=10,
                    validation_data=testset,
                    validation_steps=30)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(intent)
intent = lbl_encoder.transform(intent)

vocab_size = 1000
embedding_dim = 16
max_len = 50
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
"""
