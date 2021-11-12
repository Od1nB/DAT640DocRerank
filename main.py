import pandas as pd

from evaluate import compute_mrr
from tqdm import tqdm
from transformers import AlbertTokenizer, TFAlbertModel, AdamW
from datahandler import load_top100, load_lookup, load_qrels, load_document, create_training, create_test

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def create_model():
    # seems like tokenizer adds [CLS] to beginning and [SEP] between text
    #encoded = tokenizer(train.loc[1,"Query"],train.loc[1,"Text"],padding='max_length',truncation=True,max_length=768,return_tensors='tf')
    encoded = tokenizer(train.loc[1, "Query"] + "[SEP]" + train.loc[1, "Text"], padding='max_length', truncation=True,
                        max_length=768, return_tensors='tf')
    #hidden_size = 768
    inid = encoded['input_ids']
    mask = encoded['attention_mask']
    ttid = encoded['token_type_ids']
    print(inid)
    print(mask)
    print(ttid)
    print(embeddings([inid,mask,ttid])["pooler_output"])

    query_text = tf.keras.layers.Input(shape=(),dtype=tf.string)
    doc_text = tf.keras.layers.Input(shape=(),dtype=tf.string)
    encoded = tokenizer(query_text, doc_text, padding='max_length', truncation=True,
                        max_length=768, return_tensors='tf')


def create_cls(index,dataset):
    encoded = tokenizer(dataset.loc[index, "Query"], dataset.loc[index, "Text"], padding='max_length', truncation=True,
                        max_length=768, return_tensors='tf')

    output = embeddings([encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']])[
        "pooler_output"]
    return output.numpy()


def create_train_cls(size,dataset):
    mat = np.empty([size,768])
    y = dataset.loc[0:size-1,"Label"].to_numpy(dtype='uint8')
    for i in tqdm(range(size)):
        mat[i,:]=create_cls(i,dataset)
    return mat, y


def create_test_cls(size,dataset):
    mat = np.empty([size,768])
    for i in tqdm(range(size)):
        mat[i,:]=create_cls(i,dataset)
    return mat


def create_output(dataset,pred):
    reslst = []
    dataset['Score'] = pred
    #print(dataset.head())
    lst = dataset.values.tolist()
    lst = sorted(lst,key=lambda x: (x[0],x[5]),reverse=True)
    out = []
    line = ""
    for row in lst:
        line += str(row[0]) + " Q0 " + str(row[1]) + " " + str(row[5]) + " test1\n"
    with open("test1.txt","w") as f:
        f.write(line)


def create_classifier(train_X,train_y,test_X):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(768,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(x=train_X, y=train_y, epochs=10)
    pred = model.predict(test_X)
    return pred


if __name__ == "__main__":
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    embeddings = TFAlbertModel.from_pretrained('albert-base-v2')

    # 367012
    train = create_training(100, 500,100)

    train_X, train_y = create_train_cls(100,train)
    #print(testing)
    #print(y)

    # 5793
    testdf = create_test(100,500,5793)
    test = create_test_cls(testdf.shape[0],testdf)
    #print(test[0,:])
    #print(testdf.shape)
    scores = create_classifier(train_X,train_y,test)
    #print(len(scores))
    create_output(testdf,scores)
    #tf.keras.utils.plot_model(model)
    #print(encoded)
    #print(model(**encoded))
    #print(train.head())
    #albert_url = 'https://tfhub.dev/tensorflow/albert_en_base/2'
    #encoder = hub.KerasLayer(albert_url)
    #preprocessor_url = "https://tfhub.dev/tensorflow/albert_en_preprocess/3"
    #preprocessor = hub.KerasLayer(preprocessor_url)
