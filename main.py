from evaluate import compute_mrr
from datahandler import load_top100, load_lookup, load_qrels, load_document, create_training, create_test

if __name__ == "__main__":
    train = create_training(100, 1,100)
    #test = create_test(100,1,100)
    #print(test[0])
    #print(test[1])
    #print(len(test))
    #doc_lookup = load_lookup("data/msmarco-docs-lookup.tsv")
    # print(doc_lookup['D1272219'])
    # training data
    top100_train = load_top100("data/train/msmarco-doctrain-top100")
    # print(top100_train['174249'])
    qrels_train = load_qrels("data/train/msmarco-doctrain-qrels.tsv")
    # queries_train = load_queries("data/queries.docdev.tsv")
    # print(len(queries_train.keys()))
    # print(getsizeof(top100_train))
    # print(len(top100_train.keys()))
    # print(len(qrels_train.keys()))
    # dev data
    top100_dev = load_top100("data/dev/docdev-stopstem.xml_1.out")
    # queries_dev = load_queries("data/queries.docdev.tsv")
    qrels_dev = load_qrels("data/dev/msmarco-docdev-qrels.tsv")
    # print(load_document("data/msmarco-docs.tsv",doc_lookup['D1272219']))
    # print(top100_test['174249'])

    # top100_test = load_top100("data/test/docleaderboard-top100.tsv")
    # compute mrr of dev and train

    # print(len(top100_dev))
    # print(len(qrels_dev))
    # print(len(top100_train))
    # print(len(qrels_train))
    #print(compute_mrr(top100_dev,qrels_dev))
    #print(compute_mrr(top100_train, qrels_train))

    # print(len(queries_dev.keys()))
    # print(queries_dev.keys())
    # print(len(qrels_dev.keys()))
    # print(qrels_dev.keys())
    # print(qrels_dev["174249"])
    # print(top100_dev['174249'])
    # print(queries_dev['174249'])
    # print(len(top100_dev.keys()))
