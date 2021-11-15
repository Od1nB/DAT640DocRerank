import pandas as pd

from evaluate import compute_mrr
from tqdm import tqdm
from transformers import AlbertTokenizer, TFAlbertModel, AdamW,TFAlbertForSequenceClassification
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

# new test
def create_encoding(index,dataset):
    encoded = tokenizer(dataset.loc[index, "Query"], dataset.loc[index, "Text"], padding='max_length', truncation=True,
                        max_length=512, return_tensors='tf')
    print(encoded['input_ids'])
    print(encoded['attention_mask'])
    print(encoded['token_type_ids'])
    return encoded['input_ids'],encoded['attention_mask'], encoded['token_type_ids']


def create_train_encodings(size,dataset):
    mat = np.empty([size,512],dtype=np.int32)
    mat1 = np.empty([size, 512],dtype=np.int32)
    mat2 = np.empty([size, 512],dtype=np.int32)
    y = dataset.loc[0:size-1,"Label"].to_numpy(dtype='uint8')
    for i in tqdm(range(size)):
        mat[i,:],mat1[i,:],mat2[i,:] =create_encoding(i,dataset)
    res = {"input_ids":mat,'attention_mask':mat1,'token_type_ids':mat2}
    return res, y


def create_test_encodings(size,dataset):
    mat = np.empty([size,512],dtype=np.int32)
    mat1 = np.empty([size, 512],dtype=np.int32)
    mat2 = np.empty([size, 512],dtype=np.int32)
    for i in tqdm(range(size)):
        mat[i,:],mat1[i,:],mat2[i,:] =create_encoding(i,dataset)
    res = {"input_ids":mat,'attention_mask':mat1,'token_type_ids':mat2}
    return res


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
    for i, row in enumerate(lst):
        rank = (i%100)+1
        line += str(row[0]) + " Q0 " + str(row[1]) + " " + str(rank) + " " + str(row[5]) + " test1\n"
    with open("test1.txt","w") as f:
        f.write(line)


def create_classifier(train_X,train_y,test_X):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(768,)))
    model.add(tf.keras.layers.Dense(768, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(x=train_X, y=train_y, epochs=10)
    pred = model.predict(test_X)
    return pred


if __name__ == "__main__":
    """
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    embeddings = TFAlbertModel.from_pretrained('albert-base-v2')

    # 367012
    train = create_training(1000, 2500,10000)

    train_X, train_y = create_train_cls(train.shape[0],train)
    #print(testing)
    #print(y)

    # 5793
    testdf = create_test(1000,2500,5793)
    test = create_test_cls(testdf.shape[0],testdf)
    #print(test[0,:])
    #print(testdf.shape)
    scores = create_classifier(train_X,train_y,test)
    #print(len(scores))
    create_output(testdf,scores)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    """
    train_model = True
    train_size = 10000
    train_path = "models/trained_model"

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    if train_model:

        train = create_training(1000, 3000, train_size)
        train_X, train_y = create_train_encodings(train.shape[0], train)
        #print(train_X['input_ids'])
        #train_encoded = [tokenizer(train.loc[index, "Query"], train.loc[index, "Text"], padding='max_length', truncation=True,
        #                    max_length=768, return_tensors='tf') for index in tqdm(range(train.shape[0]))]
        #train_X = {}
        #train_X['input_ids'] = train_encoded['input_ids']
        #train_y = train['Label'].tolist()
        model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.SparseCategoricalAccuracy(),
        )
        model.fit(x=train_X, y=train_y,batch_size=16, epochs=1)
        model.save_pretrained(train_path)
    else:
        model = TFAlbertForSequenceClassification.from_pretrained(train_path)
        testdf = create_test(100, 500, 5793,False) #5793
        test = create_test_encodings(testdf.shape[0], testdf)
        scores_obj = model.predict(test)
        scores = [score[1] for score in scores_obj.logits]
        create_output(testdf, scores)
    #testdf = create_test(1000,2500,5793)
    #test = create_test_cls(testdf.shape[0],testdf)
    #scores = model.predict(test)
    #create_output(testdf, scores)

