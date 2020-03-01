# codeing: utf-8

import mxnet as mx
import numpy as np
import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock


class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, filters=[8, 10], num_conv_layers=2,
                 num_classes=2, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)
        
        with self.name_scope():
            # embedding layer
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            # pooling layer
            self.pool = gluon.nn.MaxPool1D()
            # convolutional block
            self.encoder = gluon.nn.HybridSequential()
            # output layer
            self.output = gluon.nn.HybridSequential()
            for f in filters:
                self.encoder.add(gluon.nn.Conv1D(emb_output_dim, f, activation='relu'))
                self.encoder.add(self.pool)
            self.encoder.add(gluon.nn.Flatten())

            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(0.4))
                self.output.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        encoded = self.encoder(embedded)
        return self.output(encoded)
            

class DANTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=2, prefix=None, params=None):
        super(DANTextClassifier, self).__init__(prefix=prefix, params=params)
        
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.pooler = gluon.nn.GlobalAvgPool1D()
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Dense(100, activation='relu'))
                self.encoder.add(gluon.nn.Dense(50, activation='relu'))
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(0.2))
                self.output.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        pooled = self.pooler(embedded)
        encoded = self.encoder(pooled)
        return self.output(encoded)
