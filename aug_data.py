import pandas as pd
import nltk
import random
import operator
from googletrans import Translator
from googletrans.gtoken import TokenAcquirer
from deep_translator import MicrosoftTranslator

train_file = open("archive_gavagai/is_train.json", 'r')
train_file_oos = open("archive_gavagai/oos_train.json", 'r')

data = pd.read_json(train_file)
data = data.append(pd.read_json(train_file_oos))
data = data.rename(columns={0: 'sentence', 1: 'scope'})
li = data['scope'].tolist()
l = []
for i in li:
    if i not in l:
        l.append(i)

data['scope'] = data['scope'].replace(l[:-1], 'ins')
data['scope'] = data['scope'].replace('oos', 'oos')
print(data)

sr = random.SystemRandom()

language = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "kk", "mt", "ps"]


def data_augmentation(message, language, aug_range=1):
    augmented_messages = []

    if hasattr(message, "decode"):
        message = message.decode("utf-8")

    for j in range(0, aug_range):
        new_message = ""
        target_lang = sr.choice(language)
        text = MicrosoftTranslator(api_key='91feb8bceb154ef1911258c6fdf4fa23',
                                   target=target_lang).translate(text=message)  ## Converting to random langauge for meaningful variation
        text = MicrosoftTranslator(api_key='43d34e2d59ec4770a44d4224b69c5633',
                                   target='en').translate(text=text)
        augmented_messages.append(str(text))

    return augmented_messages


intent_count = data.scope.value_counts().to_dict()
max_intent_count = max(intent_count.items(), key=operator.itemgetter(1))[1]

import numpy as np
import math
import tqdm

newdf = pd.DataFrame()
for intent, count in intent_count.items():
    count_diff = max_intent_count - count  ## Difference to fill
    multiplication_count = math.ceil(
        (count_diff) / count)  ## Multiplying a minority classes for multiplication_count times
    if (multiplication_count):
        old_message_df = pd.DataFrame()
        new_message_df = pd.DataFrame()
        for message in tqdm.tqdm(data[data["scope"] == intent]["sentence"]):
            ## Extracting existing minority class batch
            dummy1 = pd.DataFrame([message], columns=['sentence'])
            dummy1["scope"] = intent
            old_message_df = old_message_df.append(dummy1)

            ## Creating new augmented batch from existing minority class
            new_messages = data_augmentation(message, language, multiplication_count)
            dummy2 = pd.DataFrame(new_messages, columns=['sentence'])
            dummy2["scope"] = intent
            new_message_df = new_message_df.append(dummy2)

        ## Select random data points from augmented data
        new_message_df = new_message_df.take(np.random.permutation(len(new_message_df))[:count_diff])

        ## Merge existing and augmented data points
        newdf = newdf.append([old_message_df, new_message_df])
    else:
        newdf = newdf.append(data[data["scope"] == intent])

newdf.Intent.value_counts()