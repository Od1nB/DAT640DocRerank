from tqdm import tqdm
import csv
import random
import pandas as pd


def load_queries(path):
    # returns: (dict) quid to query
    res = {}
    tsv_file = open(path, "r", encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for data in read_tsv:
        res[str(data[0])] = str(data[1])
    tsv_file.close()
    return res


def load_qrels(path):
    # relevant documents to qid
    # returns: (dict) qid to docid
    res = {}
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            data = line.split(" ")
            if str(data[0]) in res.keys():
                print("test")
                res[str(data[0])].append(data[2])
            else:
                res[str(data[0])] = [data[2]]
    return res


def load_top100(path):
    # BM25 scored top 100
    # returns: (dict) qid to list of lists with docid, rank and score
    top = {}
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            data = line.split(" ")
            if str(data[0]) in top.keys():
                top[str(data[0])].append(data[2])  # ,data[3],data[4]])
            else:
                top[str(data[0])] = [data[2]]  # ,data[3], data[4]]]
    return top


def load_lookup(path):
    # returns_ (dict) docid to offset
    res = {}
    tsv_file = open(path, "r", encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for data in read_tsv:
        res[str(data[0])] = int(data[2])
    tsv_file.close()
    return res


def load_document(offset):
    # loads document at given offset
    # returns: (dict) with docid, title and body
    tsv_file = open("data/msmarco-docs.tsv", "r", encoding="utf8")
    tsv_file.seek(offset)
    document = tsv_file.readline().split("\t")
    return {"docid": document[0], "url": document[1], "title": document[2], "body": document[3]}
    print(document[0])
    print(document[2])


# title and length should be max input of
def create_training(max_length_title, max_length_body, limit):
    doc_lookup = load_lookup("data/msmarco-docs-lookup.tsv")
    top100 = load_top100("data/train/msmarco-doctrain-top100")
    qrels = load_qrels("data/train/msmarco-doctrain-qrels.tsv")
    queries = load_queries("data/train/queries.doctrain.tsv")
    random.seed(100)

    # want on format [query,text,label]
    # use the relevant doc and one none relevant for training?
    i = 0
    df = pd.DataFrame(columns=['Query','Text','Label'])
    for qid, docs in tqdm(top100.items()):
        # get positive and negative labeled docids
        doc_neg = qrels[qid][0]
        doc_pos = qrels[qid][0]
        while doc_neg == doc_pos:
            n = random.randint(0, len(docs) - 1)
            doc_neg = docs[n]

        # get text from query and doc
        query = queries[qid]
        neg_text = load_document(doc_lookup[doc_neg])
        nt = neg_text['title'][:max_length_title] + " " + neg_text['body'][:max_length_body]
        pos_text = load_document(doc_lookup[doc_pos])
        pt = pos_text['title'][:max_length_title] + " " + pos_text['body'][:max_length_body]
        df = df.append({'Query' : query, 'Text' : nt, 'Label': 0}, ignore_index=True)
        df = df.append({'Query' : query, 'Text' : pt, 'Label': 1}, ignore_index=True)
        if i > limit:
            return df
        i += 1
    return df


def create_test(max_length_title, max_length_body):
    doc_lookup = load_lookup("data/msmarco-docs-lookup.tsv")
    top100 = load_top100("data/dev/docdev-stopstem.xml_1.out")
    queries = load_queries("data/dev/queries.docdev.tsv")
    random.seed(100)
    data = []
    for qid, docs in tqdm(top100.items()):
        query = queries[qid]
        for doc in docs:
            text = load_document(doc_lookup[doc])
            t = text['title'][:max_length_title] + " " + text['body'][:max_length_body]
            data.append([query, t])
    return data
