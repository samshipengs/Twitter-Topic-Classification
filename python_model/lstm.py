import numpy as np

# TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additionally filter out WARNING, 3 to additionally filter out ERROR.
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import time

# import user defined load_data to build input data
from load_data import Data

# load data and its labels(one-hot encoded)
# i.e. classes, 1-5
'''
1 = [1, 0, 0, 0, 0]
2 = [0, 1, 0, 0, 0]
'''
file_path = '../files/'
data_file = file_path + 'data/sports-600.npy'
label_file = file_path + 'data/labels.npy'

data = np.load(data_file)
label = np.load(label_file)


# load original tweets
# ---------------------------------------------------------------------------------
sports_dic = {'basketball':1, 'hockey':2, 'baseball':3, 'tennis':4, 'volleyball':5}
sp_data = Data(sports_dic, file_path)
sp_df = sp_data.csv_df(['text']) # load data
rm_hashtags = ['#'+s for s in sports_dic.keys()]
sp_data.pre_process(sp_df, rm_list=rm_hashtags) # pre-process data
sp_df.drop(['tokenized'], axis=1, inplace=True)
# ---------------------------------------------------------------------------------


# set up lstm structure
n_classes = 5
hm_epochs = 20
batch_size = 50
chunk_size = data.shape[2]
n_chunks = data.shape[1]
rnn_size = 300

# height x width 
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# define ftn that generates batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def recurrent_neural_network(x, cv_i):
	
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])		 
	x = tf.split(x, n_chunks, 0)

	# Define a lstm cell with tensorflow
	with tf.variable_scope('cell_def'+str(cv_i)):
		lstm_cell = rnn.BasicLSTMCell(rnn_size)
	# Get lstm cell output
	with tf.variable_scope('rnn_def'+str(cv_i)):
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype =tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output

def harsh_accuracy(conf_mat):
	# conf_mat: n_classes x n_classes array
	n_c = conf_mat.shape[0]
	class_accuracy = [conf_mat[diag, diag]*1./sum(conf_mat[diag, :]) for diag in range(n_c)]
	print_class_accuracy = [float(str(i)[:6]) for i in class_accuracy]
	print "Accuracy per class is: ", print_class_accuracy
	return np.mean(class_accuracy)


def train_neural_network(x, data_train, y_train, data_val, y_val, cv_i, verbose=True):
	prediction = recurrent_neural_network(x, cv_i)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	# learning_rate = 0.001 default
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for batch in iterate_minibatches(data_train, y_train, 100, shuffle=True):
				epoch_x, epoch_y = batch
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c 
			if verbose:	
				print 'Epoch', epoch, 'completed out of', hm_epochs, epoch_loss


		# correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		
		# prediction
		output_prediction=sess.run(tf.argmax(prediction, 1), feed_dict={x: data_val})
		# prediction_result = output_prediction.eval({x: data_val, y:y_val})

		# confusion matrix
		conf_mat = tf.contrib.metrics.confusion_matrix(tf.argmax(y, 1), tf.argmax(prediction, 1), num_classes=n_classes)
		# print('Accuracy:', accuracy.eval({x:X_val, y:y_val}))
		test_cm = conf_mat.eval({x:data_val, y:y_val})
		print 'Confusion matrix:\n', test_cm
		print '(Harsh) Accuracy: {0:.2f}'.format(harsh_accuracy(test_cm))

		# return predictions
		return output_prediction + 1 # classes are in index, plus 1 to get back to labels

# train_neural_network(x)
def train_cv(x, data, label, n_cv = 1, verbose=True):
	# stratified split
	sss = StratifiedShuffleSplit(n_splits=n_cv, test_size=0.25, random_state=0)
	# X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.25)
	# print 'X_train shape is: ', X_train.shape
	# print 'y_train shape is: ', y_train.shape
	n = 1
	for train_index, val_index in sss.split(data, label):
		print "[CV {}]".format(n)

		X_train, X_val = data[train_index], data[val_index]
   		y_train, y_val = label[train_index], label[val_index]

   		# extract original data to do a end-to-end check on predictions
   		df_n = sp_df.loc[val_index, :].copy()

   		t_start = time.time()
   		prediction_n = train_neural_network(x, X_train, y_train, X_val, y_val, n, verbose=verbose)
   		t_end = time.time()

   		# write prediction to original df_n
   		# --------------------------------------------------------
   		df_n['prediction'] = prediction_n
   		df_n.to_csv(file_path+'data/predictions.csv', index=False)
   		# --------------------------------------------------------

   		print "Time took {0:.4f} min\n".format((t_end-t_start)/60.)
   		n += 1

# train_cv(x, data, label, n_cv=5, verbose=False)
train_cv(x, data, label, n_cv=1, verbose=False)

# result
'''
[CV 1]
Confusion matrix:
[[2335   35   28   64   27]
 [  53 2356   52   78   42]
 [  24   48 2289   56   16]
 [  44   56   35 2342   32]
 [  23   42   28   72 1012]]
Accuracy per class is:  [0.9381, 0.9128, 0.9408, 0.9334, 0.8598]
(Harsh) Accuracy: 0.92
Time took 4.6391 min

[CV 2]
Confusion matrix:
[[2427   52   26   37   36]
 [  47 2378   37   29   34]
 [  66   50 2281   31   18]
 [  83   61   21 2239   78]
 [  40   64   15   30 1009]]
Accuracy per class is:  [0.9414, 0.9417, 0.9325, 0.902, 0.8713]
(Harsh) Accuracy: 0.92
Time took 4.6155 min

[CV 3]
Confusion matrix:
[[2387   46   24   90   20]
 [  43 2278   30   76   47]
 [  40   51 2293   51   18]
 [  26   44   21 2392   34]
 [  39   38   18   72 1011]]
Accuracy per class is:  [0.9298, 0.9207, 0.9347, 0.9503, 0.8582]
(Harsh) Accuracy: 0.92
Time took 4.6091 min

[CV 4]
Confusion matrix:
[[2386   58   17   46   15]
 [  64 2311   27   63   21]
 [  63   42 2339   39   14]
 [  68   44   27 2291   33]
 [  76   72   25   92  956]]
Accuracy per class is:  [0.946, 0.9296, 0.9367, 0.9301, 0.7829]
(Harsh) Accuracy: 0.91
Time took 4.4747 min

[CV 5]
Confusion matrix:
[[2239   38   46   78   39]
 [  28 2295   47   55   49]
 [  20   36 2442   43   13]
 [  33   35   33 2383   42]
 [  27   42   18   74 1034]]
Accuracy per class is:  [0.9176, 0.9276, 0.9561, 0.9433, 0.8652]
(Harsh) Accuracy: 0.92
Time took 4.4542 min
'''