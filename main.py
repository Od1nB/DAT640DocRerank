
from tqdm import tqdm
from datahandler import create_training, create_test
from transformers import AlbertTokenizer, logging, TFAlbertForSequenceClassification

import pandas as pd
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
import official.nlp.optimization
from official import nlp


logging.set_verbosity_error()

def create_encoding(index,dataset):
    encoded = tokenizer(dataset.loc[index, "Query"], dataset.loc[index, "Text"], padding='max_length',
                        truncation=True, max_length=256, return_tensors='tf')
    return encoded['input_ids'],encoded['attention_mask'], encoded['token_type_ids']

def create_train_encodings(size,dataset):
    mat = np.empty([size,256],dtype=np.int32)
    mat1 = np.empty([size, 256],dtype=np.int32)
    mat2 = np.empty([size, 256],dtype=np.int32)
    y = dataset.loc[0:size-1,"Label"].to_numpy(dtype='uint8')
    for i in tqdm(range(size)):
        mat[i,:],mat1[i,:],mat2[i,:] =create_encoding(i,dataset)
    res = {"input_ids":mat,'attention_mask':mat1,'token_type_ids':mat2}
    return res, y

def create_test_encodings(size,dataset):
    mat = np.empty([size,256],dtype=np.int32)
    mat1 = np.empty([size, 256],dtype=np.int32)
    mat2 = np.empty([size, 256],dtype=np.int32)
    for i in tqdm(range(size)):
        mat[i,:],mat1[i,:],mat2[i,:] =create_encoding(i,dataset)
    res = {"input_ids":mat,'attention_mask':mat1,'token_type_ids':mat2}
    return res

def create_output(dataset,pred):
    dataset['Score'] = pred
    lst = dataset.values.tolist()
    lst = sorted(lst,key=lambda x: (x[0],x[5]),reverse=True)
    line = ""
    for i, row in enumerate(lst):
        rank = (i%100)+1
        line += str(row[0]) + " Q0 " + str(row[1]) + " " + str(rank) + " " + str(row[5]) + " test1\n"
    with open("test1.txt","w") as f:
        f.write(line)

if __name__ == "__main__":
    train_model = False
    train_size = 100000 
    train_path = "models/model_folder"

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    if train_model:
        train = create_training(500, 2500, train_size)
        train_X, train_y = create_train_encodings(train.shape[0], train)
        model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
        optimizer = nlp.optimization.create_optimizer(
            1e-7, num_train_steps=train_size, num_warmup_steps=600)
        model.compile(
            optimizer=optimizer,
            loss=tfa.losses.TripletSemiHardLoss(),
            metrics=['accuracy'],
        )
        model.fit(x=train_X, y=train_y,batch_size=16, epochs=1)
        model.save_pretrained(train_path)
    else:
        model = TFAlbertForSequenceClassification.from_pretrained(train_path)
        testdf = create_test(500, 2500, 5793, False)
        test = create_test_encodings(testdf.shape[0], testdf)
        scores_obj = model.predict(test)
        scores = [score[1] for score in scores_obj.logits]
        create_output(testdf, scores)
