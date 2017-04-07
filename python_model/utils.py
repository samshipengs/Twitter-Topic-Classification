# Provide save model and load model and make predictions function
import numpy as np
import lasagne
import nltk

def save_network(filepath, filename, network):
	print "Saving {}.npz to disk ...".format(filename)
	np.savez(filepath + 'model/' + filename, *lasagne.layers.get_all_param_values(network))
	print "Done saving."


def load_network(filepath, filename):
	with np.load(filepath + 'model/' + filename) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	return param_values

def most_freq(df, class_label, top=10, plot=False):
	# df should have a colum cotaining tokenized words and a class(prediction) column
	# class_label here is numeric value
	df_select = df[df['prediction']==class_label]
	word_frequency = nltk.FreqDist(i for w in df_select['tokenized'] for i in w)
	if plot:
		word_frequency.plot(top)
	return word_frequency.most_common(top)

def look_up(df_origin, df_test, word, class_label, look=10):
	# df_origin contains raw tweet text, df_test is pre-processed
	n = 0
	for i in range(df_origin.shape[0]):
		if word in list(df_test['tokenized'])[i]:
			if list(df_test['prediction'])[i] == class_label:
				print list(df_origin['text'])[i]
				n += 1
		if n == look:
			break
