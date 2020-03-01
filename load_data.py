# coding: utf-8

import re
import io
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import gluonnlp as nlp
import mxnet as mx
from nltk.corpus import stopwords


def load_dataset(train_file, val_file, test_file, tokenizr, max_length=32):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array, tr_labels = load_tsv_to_array(train_file)
    val_array, vl_labels = load_tsv_to_array(val_file)
    test_array, ts_labels = load_tsv_to_array(test_file)
    
    vocabulary = build_vocabulary(train_array, val_array, test_array, tokenizr)
    train_dataset = preprocess_dataset(train_array, vocabulary, max_length, tokenizr)
    val_dataset = preprocess_dataset(val_array, vocabulary, max_length, tokenizr)
    test_dataset = preprocess_dataset(test_array, vocabulary, max_length, tokenizr)
    # OPTION 1: Create numpy or NDArrays here up front and return for train, val and test datasets

    # OPTION 2: Keep data as python data and return a transform object (BasicTransform below) that
    # maps to NDArrays on the fly

    return vocabulary, train_dataset, val_dataset, test_dataset


def load_tsv_to_array(infile):
    tweets = []
    labels = ['Relevant', 'Not Relevant']
    # set up stopwords
    engstop = stopwords.words('english')
    with open(infile, encoding='utf8') as f:
        for line in f:
            splits = line.split('\t')
            lab = splits[1]
            tweet = splits[2][:-1]

            cut_tweet = ' '.join([w for w in tweet.split() if w not in engstop])
            line_tweet = (lab, cut_tweet)

            if lab in labels:
                tweets.append(line_tweet)
    return tweets, labels


def build_vocabulary(tr_array, val_array, tst_array, tkzr):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []

    for i, instance in enumerate(tr_array):
        # i, label, text = instance
        text = instance[1]
        tokens = tkzr.tokenize(text)
        all_tokens.extend(tokens)
    for i, instance in enumerate(val_array):
        # id_num, label, text = instance
        text = instance[1]
        tokens = tkzr.tokenize(text)
        # all_tokens.extend(tokens)
    for i, instance in enumerate(tst_array):
        # id_num, label, text = instance
        text = instance[1]
        tokens = tkzr.tokenize(text)
        # all_tokens.extend(tokens)
    
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    
    # attach an embedding so that if not done in the training file, there will be one
    glove = nlp.embedding.create('glove', source='glove.twitter.27B.100d', unknown_token='<unk>')
    vocab.set_embedding(glove)
    # use built-in unknown vector (present in glove, which I use)
    return vocab


def _preprocess(x, vocab, tkzr, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    labels = ['Relevant', 'Not Relevant']
    # initialize transform object for later on
    tran = BasicTransform(labels)
    label = x[0]
    text = x[1]
    text_tokens = tkzr.tokenize(text)
    data = vocab[text_tokens]
    # do the 0-padding myself
    if len(data) > max_len:
        norm_data = data[:max_len+1]
    elif len(data) < max_len:
        while len(data) < max_len:
            data.append(0)
        norm_data = data
    else:
        norm_data = data
    # transform data and return correct arrays for models
    return tran(label, norm_data)


# basically as given
def preprocess_dataset(dataset, vocab, max_len, tkzr):
    preprocessed_dataset = [ _preprocess(x, vocab, tkzr, max_len) for x in dataset ]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, labels, max_len=64):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
    
    def __call__(self, label, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        return mx.nd.array(padded_data, dtype='int32'), mx.nd.array([label_id], dtype='int32')


if __name__ == '__main__':
    tokenizer = TweetTokenizer()
    load_dataset('data/disaster_tweets_train.tsv', 'data/disaster_tweets_val.tsv',
                 'data/disaster_tweets_test.tsv', tokenizer)
