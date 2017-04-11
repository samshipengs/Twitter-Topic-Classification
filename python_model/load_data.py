#!/usr/bin/python
# This script loads data, convert them to vectors and save them to disk.

# load libraries
import pandas as pd
import numpy as np
import re
from gensim.models import word2vec
import logging
import nltk
from collections import Counter
import itertools
from nltk.corpus import stopwords
import os.path
import random

from sklearn.preprocessing import StandardScaler

class Data:
	def __init__(self, file_name, file_path):
		self.file_name = file_name
		self.FILE_PATH = file_path
		self.data_path = file_path+'data/'

	# def function to load data from json to dataframe
	def json_df(self, json_fields):
	    print "Loading json: " + self.file_name + " ..."
	    data_df = pd.read_json(self.data_path + self.file_name, lines=True)
	    # we only take the 'text' column
	    drop_columns = list(data_df.columns)
	    # drop_columns.remove('text')
	    for i in json_fields:
	    	drop_columns.remove(i)
	    data_df.drop(drop_columns, axis = 1, inplace = True)
	    data_df.dropna(axis=0, inplace=True) # drop na rows
	    print "Done loading json file to dataframe."
	    return data_df

	def csv_df(self, csv_fields):
		if len(self.file_name) > 1:
			print "Loading csv: " 
			for i in self.file_name.keys():
				print '  ' + i
			df_list = []
			for k in self.file_name:
				df_k = pd.read_csv(self.data_path + k + '.csv')
				df_k = df_k[csv_fields]
				df_k.dropna(axis=0, inplace=True) # drop na rows
				df_k.loc[:,'class'] = self.file_name[k]
				df_list.append(df_k)
			return pd.concat(df_list).reset_index(drop=True)
		else:	
			print "Loading csv: " + self.file_name + " ..."
			data_df = pd.read_csv(self.data_path + self.file_name)
			data_df =  data_df[csv_fields]
			data_df.dropna(axis=0, inplace=True) # drop na rows
			return data_df

	# possibly remove bots, but it requires a decision on the score of results, see blow
	# so far not used here.
	# def remove_bots(self, df):
	# 	# A username can only contain alphanumeric characters (letters A-Z, numbers 0-9) with the exception of underscores
	# 	import botornot

	# 	twitter_app_auth = {
	# 	    'consumer_key': 'xxxxxxxx',
	# 	    'consumer_secret': 'xxxxxxxxxx',
	# 	    'access_token': 'xxxxxxxxx',
	# 	    'access_token_secret': 'xxxxxxxxxxx',
	# 	  }
	# 	bon = botornot.BotOrNot(**twitter_app_auth)

	# 	# Check a single account
	# 	result = bon.check_account('@clayadavis')

	# 	# Check a sequence of accounts
	# 	accounts = ['@clayadavis', '@onurvarol', '@jabawack']
	# 	results = list(bon.check_accounts_in(accounts))

	# pre-processing text
	def pre_process(self, df, rm_list=None):
		print("Note: pre-process changes the dataframe inplace.")
		if rm_list:
			print "Removing ", rm_list
			for rm in rm_list:
				df['text'] = df['text'].apply(lambda row: re.sub(rm, ' ', row, flags=re.IGNORECASE))
		# remove new line char
		df['text'].replace(regex=True,inplace=True,to_replace='(\\n|\\r|\\r\\n)',value=' ')
		# remove https links
		df['text'].replace(regex=True,inplace=True,to_replace='(http|https):\/\/[^(\s|\b)]+',value=' ')
		# remove user name 
		# do not remove user name for topic(sports) classification 
		# df['text'].replace(regex=True,inplace=True,to_replace=r'@\w+',value=r'')
		# remove non-alphabet, this includes number and punctuation
		df['text'].replace(regex=True,inplace=True,to_replace='[^a-zA-Z\s]',value=' ')
		# tokenize each tweets to form sentences.
		df['tokenized'] = df['text'].apply(lambda row: nltk.word_tokenize(row.lower()))
		# remove stop words
		stop_words = stopwords.words('english')
		add_stop_words = ['amp', 'rt']
		stop_words += add_stop_words
		# also remove english names
		# last_names = [x.lower() for x in np.loadtxt(self.FILE_PATH+"last_names.txt", usecols=0, dtype=str)[:5000]]
		# stop_words += last_names
		# first_names = [x.lower() for x in np.loadtxt(self.FILE_PATH+"first_names.txt", usecols=0, dtype=str)]
		# stop_words += first_names
		# print "sample stopping words: ", stop_words[:5]
		df['tokenized'] = df['tokenized'].apply(lambda x: [item for item in x if item not in stop_words])
		df['text'] = df['tokenized'].apply(lambda row: ' '.join(row))

	# now let us bring in the wordvec trained using text8 dataset
	def build_wordvec(self, size = 200, verbose=True):
		model_name='tweets' + str(size) + '.model.bin'
		self.vec_size = size
		sentences = word2vec.Text8Corpus(self.FILE_PATH + 'data/text8') # use text 8
		model_path = self.FILE_PATH + 'wordvec/' + model_name
		if os.path.isfile(model_path):
			print "Loading existing model {} ...".format(model_name)
			model = word2vec.Word2Vec.load(model_path)
		else:
			if verbose:
				logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
			print "Training for {} ...".format(model_name)
			model = word2vec.Word2Vec(sentences, size=self.vec_size, sg=1, workers=4)
			model.save(model_path)
			# If you finished training a model (no more updates, only querying), you can do
			model.init_sims(replace=True)
		print "Done building."
		return model

	# transform our tweets using vector representation
	# first find the max length since that decides the padding
	def max_len(self, df):
		df['size'] = df['tokenized'].apply(lambda x: len(x))
		print "max sentence length is: ", df['size'].max()
		return df['size'].max() 

	# initialize empty arry to fill with vector repsentation
	def convert2vec(self, df, max_length, model, name='default'):
		file_name = self.FILE_PATH + name
		if os.path.isfile(file_name + '.npy'):
			print "npy already exists, loading ..."
			tweet_vecs = np.load(file_name + '.npy')
			print "Done loading npy file."
			return tweet_vecs
		else:
			tweet_tokens = df['tokenized'].values
			n = tweet_tokens.shape[0]
			m = max_length
			n_absent = 0
			tweet_vecs = np.zeros((n,m,self.vec_size))
			vocabs = model.wv.vocab.keys()
			for i in range(n):
				if i%2000 == 0:
					print ">>> " + str(i) + " tweets converted ..."
				token_i = [x for x in tweet_tokens[i] if x in vocabs]
				m_i = len(token_i)

				if m_i == 0:
				    n_absent += 1
				else:
					diff_i = abs(m_i - m)
					vecs_i = model[token_i]
					tweet_vecs[i] = np.lib.pad(vecs_i, ((0,diff_i),(0,0)), 'constant', constant_values=0)
			print "Total {} not in vocab.".format(n_absent)
			print "Done converting tweets to vec!"	
			return tweet_vecs

	def standarize(self, tweet_vecs):
		# tweet_vec: example dim: (8561, 19, 600)
		n1, n2, n3 = tweet_vecs.shape
		tweet_vecs = tweet_vecs.reshape(n1, n2*n3)
		tweet_vecs = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(tweet_vecs)
		tweet_vecs = tweet_vecs.reshape(n1, n2, n3)
		return tweet_vecs

	# save tweet_vecs to disk in npy
	def save_vec(self, tweet_vecs, name='default'):
		file_name = self.FILE_PATH + name
		if os.path.isfile(file_name + '.npy') and os.path.isfile(file_name + '.npz'):
			print "npy already exists."
		else:
			np.save(file_name, tweet_vecs)
			np.savez(file_name, tweet_vecs)
			print "Saved {} to disk.".format(name)
	
	# drop rows whose col equals to certain value, this is used to drop neutral classes
	def drop_value(self, df, col, value):
		try:
			df_dropped = df.drop(df[df[col]==value].index)
			df_clean = df_dropped.reset_index(drop=True)

			print "Dropped {} on column {}".format(value, col)
			return df_clean
		except :
			raise Exception("droping does not work might be column intented to drop does not exist.")

	# replace a categorical value column to numeric column based on a given dictionary
	def cat2num(self, df, col1, value_dict, col2=None):
		# class_label = {'positive': 1, 'negative': 2}
		col2 = col2 or col1 # default remain the same name
		print col2
		df[col2] = df[col1].apply(lambda x: value_dict[x])
		if col2 != col1:
			df.drop(col1, inplace=True, axis=1)
		print "Done converting categorical to numeric, this changes df."
		return df

	# convert numberic value back to categorical value based on a given dictionary	
	def num2cat(self, df, col1, value_dict, col2=None):
		# class_label = {1:'positive', 2: 'neutral', 3: 'negative'}
		col2 = col2 or col1 # default remain the same name
		print col2
		df[col2] = df[col1].apply(lambda x: value_dict[x])
		if col2 != col1:
			df.drop(col1, inplace=True, axis=1)
		print "Done converting numeric to categorical, this changes df."
		return df
	
	# balance classes, the method used here is just drop the most number of classes to the 
	# second most numer of class
	def balance_class(self, df, random_state=42):
		class_counts = df['class'].value_counts()
		n_top1 = class_counts.values[0]
		n_top2 = class_counts.values[1]
		random.seed(random_state)
		drop_n_top1 = random.sample(range(n_top1), n_top1-n_top2) # sample without replacement
		n_top1_index = df[df['class']==class_counts.index[0]].index.values
		df.drop(n_top1_index[drop_n_top1], axis=0, inplace=True)
		df = df.reset_index(drop=True)
		return df