import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit, KFold

from load_data import Data
# from tensorflow.examples.tutorials.mnist import input_data

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

X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2)

print X_train.shape
print y_train.shape



hm_epochs = 20
batch_size = 50
chunk_size = D
n_chunks = M
rnn_size = 256

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


def recurrent_neural_network(x):
	
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])		 
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype =tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output

def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	# learning_rate = 0.001 default
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# for _ in range(int(mnist.train.num_examples/batch_size)):
			# 	epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			# 	# print(epoch_x.shape, epoch_y.shape)
			# 	epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

			# 	_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
			# 	epoch_loss += c 
			# print('Epoch', epoch, 'completed out of', hm_epochs, epoch_loss)
			for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
				epoch_x, epoch_y = batch
				# print(epoch_x.shape, epoch_y.shape)
				# epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c 
			print('Epoch', epoch, 'completed out of', hm_epochs, epoch_loss)


		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:X_val, y:y_val}))

train_neural_network(x)