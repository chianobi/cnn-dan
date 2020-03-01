# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from model import CNNTextClassifier, DANTextClassifier
from load_data import load_dataset, BasicTransform
from nltk.tokenize import TweetTokenizer


parser = argparse.ArgumentParser(description="CNN for text classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data',
                    default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=5, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.005)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=32)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d',
                    help='Pre-trained embedding source name')
parser.add_argument('--fix_embedding', action='store_true', help='Fix embedding vectors instead of fine-tuning them')

args = parser.parse_args()

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
# OR if the output dense layer is just a single value (rather than 2), do:
# loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()


def train_classifier(vocabulary, data_train, data_val, data_test, ctx=mx.cpu()):

    # set up the data loaders for each data source
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape
    print(emb_output_dim)

    model = CNNTextClassifier(emb_input_dim, emb_output_dim)

    model.initialize(ctx=ctx)  # initialize model parameters on the context ctx
    model.embedding.weight.set_data(vocab.embedding.idx_to_vec) # set the embedding layer parameters to the pre-trained embedding in the vocabulary

    if args.fix_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    model.hybridize()   # OPTIONAL for efficiency - perhaps easier to comment this out during debugging
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        epoch_cum_loss = 0
        for i, tup in enumerate(train_dataloader):
            data = tup[0].as_in_context(ctx)
            label = tup[1].as_in_context(ctx)
            with autograd.record():
                output = model(data)  # should have shape (batch_size,)
                l = loss_fn(output, label).mean()  # get the average loss over the batch
            l.backward()
            trainer.step(label.shape[0]) # update weights
            epoch_cum_loss += l.asscalar()  # needed to convert mx.nd.array value back to Python float
        # show us the results on train set (which hopefully approach perfection)
        train_accuracy = evaluate(model, train_dataloader)

        val_accuracy = evaluate(model, val_dataloader)
        # display and/or collect validation accuracies after each training epoch
        print('epoch: ' + str(epoch) + '\t train loss: ' + str(epoch_cum_loss) + '\t val score: ' + str(val_accuracy))
    # show score on the test set
    test_accuracy = evaluate(model, test_dataloader, test=True)
        

def evaluate(model, dataloader, ctx=mx.cpu(), test=False):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    total_correct = 0
    total = 0
    labels = []  # store the ground truth labels
    scores = []  # store the predictions/scores from the model
    raw_scores = []
    for i, (data, label) in enumerate(dataloader):
        out = model(data)
        #  Model inference here:
        #  ...
        ##
        #  NOTE: You'll need to get your predictions from the output
        #        This will vary depending on whether you used a single output with sigmoid or two outputs with softmax
        #        For case of a single output, you'll need to apply a sigmoid - e.g. mx.nd.sigmoid(out)
        #        For two outputs, you'll need to apply softmax - e.g. mx.nd.softmax(out)
        predict = mx.nd.softmax(out)
        prediction = np.argmax(predict, axis=1)
        #  You'll then need to go over each item in the batch (or use array ops) as:
        for j in range(out.shape[0]):   # out.shape[0] refers to the batch size
            lab = int(label[j].asscalar())
            labels.append(lab)
            # gather predictions for each item here
            raw_scores.append(predict[j][1].asscalar())
            inference = int(prediction[j].asscalar())
            scores.append(inference)
        for r, p in zip(labels, scores):
            if r == p:
                total_correct += 1
            total += 1
    acc = total_correct / float(total)
    print(metrics.classification_report(labels, scores, labels=[0, 1]))
    aps = metrics.average_precision_score(labels, scores)
    # plot precision recall curve for test set!
    if test:
        prec, rec, thresh = metrics.precision_recall_curve(labels, raw_scores)
        print(prec)
        print(rec)
        print(thresh)
        plt.figure()
        lw = 2
        plt.plot(rec, prec, color='darkorange',
                 lw=lw, label='precision-recall curve (area = %0.2f)' % aps)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.55, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

    return acc
    

if __name__ == '__main__':
    tokenizer = TweetTokenizer()
    # load the vocab and datasets (train, val, test)
    vocab, train_dataset, val_dataset, test_dataset = load_dataset('data/disaster_tweets_train.tsv',
                                                                   'data/disaster_tweets_val.tsv',
                                                                   'data/disaster_tweets_test.tsv',
                                                                   tokenizer)

    # get the pre-trained word embedding
    # NOTE: running nlp.embedding.list_sources() will give a list of available pre-trained embeddings
    glove_twitter = nlp.embedding.create('glove', source='glove.twitter.27B.50d')
    vocab.set_embedding(glove_twitter)

    ctx = mx.cpu()  # or mx.gpu(N) if GPU device N is available
    
    train_classifier(vocab, train_dataset, val_dataset, test_dataset, ctx)
