# DAT640DocRerank
## Project description:
> MS Marco Document re-ranking [MS MARCO](https://microsoft.github.io/msmarco/)
>  * Given a candidate top-100 document as retrieved by BM25, re-rank documents by relevance. This task has also run as part of the TREC 2019 Deep Learning track.
>  * Document collection: MS MARCO document corpus (3.2M documents, 22GB)
>  * Resources
>    * [Task GitHub repo](https://github.com/microsoft/MSMARCO-Document-Ranking)
>    * [Relevance judgments for evaluation topics](https://trec.nist.gov/data/cast/2019qrels.txt)
>    * [TREC 2019 Deep Learning track overview](https://arxiv.org/abs/2003.07820)

### Dataset

Dataset: [here](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019), under subsection Document ranking dataset

### Data folder structure

data folder should have subfolders dev, test and train where the data for each category is stored. Document collection and document lookup file should be in the main data folder.

### Install Guide TensorFlow GPU
Guide to installing tensorflow for GPU processing: [link](https://www.tensorflow.org/install/gpu)
Just seems like we to install the required drivers and add them to path, for windows use (NB: seems like we need NVIDIA GPU)
or in this markdown [GPUINSTALL](GPUINSTALL.md)


### Albert resources

Fine-tuning Albert (google colab): [link](https://colab.research.google.com/github/NadirEM/nlp-notebooks/blob/master/Fine_tune_ALBERT_sentence_pair_classification.ipynb)

#### Albert guide (classifies imdb reviews to be positive or negative):

Article: [here](https://analyticsindiamag.com/complete-guide-to-albert-a-lite-bertwith-python-code/)

Code:[here](https://colab.research.google.com/drive/1PQ-tpKUHoxNSR-gmPJpYxpiH8JZNuMjg?usp=sharing)

### Todo:
- Find best way to combine title and body of document (should it be title+ " " + body, or just body?)

### Questions
- Find out how test data works, multiple documents are given the same rank.
- What is the "Relevance judgments for evaluation topics" link?, qid does not match qid from dataset
