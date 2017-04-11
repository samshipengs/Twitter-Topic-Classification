# This file makes predictions on incoming data and perform basic analysis.
import pandas as pd

import theano
import theano.tensor as T
from lasagne import layers

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from sklearn.externals import joblib

import nltk
from load_data import Data
from utils import *

import sys
import os.path

# define cnn sructure
def build_cnn(M, D, num_epochs=20):
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, M, D),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),  
        conv2d1_stride=1,
        conv2d1_pad=1,
        conv2d1_untie_biases=True,
        # layer maxpool1
        maxpool1_pool_size=(2, 2),    
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d2_stride=1,
        conv2d2_pad=1,
        conv2d2_untie_biases=True,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,    
        # dense
        dense_num_units=1000,
        dense_nonlinearity=lasagne.nonlinearities.rectify,    
        # dropout2
        dropout2_p=0.5,    
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=5,

        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,
        # train options
        train_split = TrainSplit(0.25, stratify=True),
        batch_iterator_train = BatchIterator(batch_size=50),
        batch_iterator_test = BatchIterator(batch_size=50),
        max_epochs=num_epochs,
        verbose=1,
        )
    return net1

# def feed forward pass ftns until dense layer to generate features for svm
def generate_features(net):
    dense_layer = layers.get_output(net.layers_['dense'], deterministic=True)
    output_layer = layers.get_output(net.layers_['output'], deterministic=True)
    input_var = net.layers_['input'].input_var

    f_output = theano.function([input_var], output_layer)
    f_dense = theano.function([input_var], dense_layer)
    
    return f_dense, f_output

# convert
def extract_features(net, input_data):
    print "Extracting ... "
    # input_data: n0, 1, n2, n3
    n = input_data.shape
    f_dense = generate_features(net)[0]
    return [f_dense(i.reshape(1,1,n[2],n[3])).flatten() for i in input_data]


# def Prediction class used in making predictions on test data
class Prediction:
	def __init__(self, file_path, file_name, max_len_train):
		self.file_path = file_path
		self.file_name = file_name
		self.max_len_train = max_len_train # length of sentence from training

	def prepare_data(self, data_fields, wv_size=600):
		test_data = Data(self.file_name, self.file_path)
		test_df = test_data.csv_df(data_fields)
		# make a copy of the original tweets for later use
		original_df = test_df.copy()

		# pre-process data(same as how we trained)
		test_data.pre_process(test_df) 

		# then convert using word2vec
		model = test_data.build_wordvec(size=wv_size, verbose=False)
		# take a look of the max_len of testing. although we still have to use max_len from train
		max_len_test = test_data.max_len(test_df)
		data = test_data.convert2vec(test_df, self.max_len_train, model, name='test_'+self.file_name)
		test_data.save_vec(data, name='test_'+self.file_name)

		self.data = data
		self.test_data = test_data
		self.test_df = test_df
		self.original_df = original_df
		print ">>>Done preparing data.<<<\n"

	def make_prediction(self, data, verbose=True):
		N, M, D = data.shape
		if verbose:
			print "N, M, D:", N, M, D
		data = data.reshape(-1, 1, M, D).astype(theano.config.floatX) # theano needs this way

		cnn = build_cnn(M, D)
		model_file = self.file_path+'model/nn_cnn'
		cnn.load_params_from(model_file)

		extract_data = extract_features(cnn, data)
		clf = joblib.load(self.file_path+'model/svm-final.pkl')
		test_pred = clf.predict(extract_data)

		return test_pred + 1

	def get_result(self, n_preview=10, n_top = 20, name = 'default', verbose=True):

		# get predictions
		test_predictions = self.make_prediction(self.data, verbose)
		### Take a look at the predictions with raw tweets
		self.test_df['prediction'] = test_predictions
		# lets take a look of the 
		if verbose:
			print self.test_df['prediction'].value_counts()

		# write to original dataframe
		self.original_df['prediction'] = test_predictions
		# convert numeric prediction to categorical
		class_label = {1:'positive', 2: 'neutral', 3: 'negative'}
		self.original_df = self.test_data.num2cat(self.original_df, 'prediction', class_label)
		if verbose:
			# take a quick look at the prediction and its corresponding tweet
			for i in range(n_preview):
				print self.original_df.values[i,]

		# save to csv
		self.original_df.to_csv(name+'.csv', index=False)

		# look at most frequent words in different groups
		print "===Positive==="
		print most_freq(self.test_df, 1, top=n_top, plot=True)
		print "===Neutral==="
		print most_freq(self.test_df, 2, plot=False)
		print "===Negative==="
		print most_freq(self.test_df, 3, top=n_top, plot=True)

	def check(self, word, sentiment, n_view=10):
		# we can take a look of the tweets where the frequent word is mentioned, e.g. lookup 'flight' in negative tweets
		look_up(self.original_df, self.test_df, word, sentiment, look=n_view)