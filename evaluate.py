
def compute_mrr(computed_documents,ground_truth):
    # computed_documents (dict) qid to document ranks
    # ground_truth (dict) qid to document ranks
    score = 0
    for qid, ranks in computed_documents.items():
        relevant = ground_truth[qid]
        for i in range(len(ranks)):
            if ranks[i] in relevant:
                score += 1/(i+1)
                break
    score = score/len(ground_truth)
    return score



def store_result():
    pass