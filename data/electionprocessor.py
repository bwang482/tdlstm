import os
import time
import re
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
from gensim import utils
from twtokenize import tokenize
import util
from sklearn.cross_validation import train_test_split
from ftfy import fix_text

class streamtwElec(object):
    def __init__(self, dirname):
        self.i=0
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = line.split('\t')
                id = line[1]
                if line[2] == 'positive':
                    sent = 1
                elif line[2] == 'negative':
                    sent = -1
                elif line[2] == 'neutral':
                    sent = 0
                senti = sent + 2
                target = line[3].lower().strip()
                location = line[4]
                tw = line[-1].lower().strip()
                tw = fix_text(tw.decode('utf-8'))
                range = []
                p = re.compile(r'(?<!\w)({0})(?!\w)'.format(target))
                for m in p.finditer(tw.lower()):
                      range.append([m.start(),m.start()+len(m.group())])
                if location != 'nan':
                   cc = 0
                   for a, b in enumerate(range):
                       if b[0]-1 <= int(location) <= b[1]+4:
                          wh = a
                          cc=1
                   if cc==0:
                       wh = 'nan'
                else:
                    wh = location  
                if wh == 'nan':
                    tw=tw.replace(target,' '+target+' ')
                    tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
                    tw=tw.replace(target,' '+'_'.join(target.split())+' ')  
                else:
                    try:
                        r = range[wh]
                    except:
                        print "Error at processing election data; at line 118 process_data.py!"
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+2].replace(target, ' '+target+' ') + tw[r[1]+2:]
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+4].replace(''.join(target.split()),' '+'_'.join(target.split())+' ') + tw[r[1]+4:]
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+6].replace(target,' '+'_'.join(target.split())+' ') + tw[r[1]+6:]
              
                tweet=tokenize(tw)
                yield (tweet,'_'.join(target.split()),senti,id,wh)


class ElectionData:
	def __init__(self, batch_size, dynamic_padding=False, preprocessing=False, embedding=True, saved=False, max_length=None):
		train = ElectionData.read_data('data/election-data/training/')
		test = ElectionData.read_data('data/election-data/testing/')
		self.batch_size = batch_size
		self.dynamic_padding = dynamic_padding
		self.train_tweets, self.train_targets, self.train_y = zip(*train)
		self.test_tweets, self.test_targets, self.test_y = zip(*test)

		self.train_left_tweets = [ElectionData.split_tweet(self.train_tweets[i], self.train_targets[i])[0] for i in range(len(self.train_tweets))]
		self.train_right_tweets = [ElectionData.split_tweet(self.train_tweets[i], self.train_targets[i])[1] for i in range(len(self.train_tweets))]
		self.test_left_tweets = [ElectionData.split_tweet(self.test_tweets[i], self.test_targets[i])[0] for i in range(len(self.test_tweets))]
		self.test_right_tweets = [ElectionData.split_tweet(self.test_tweets[i], self.test_targets[i])[1] for i in range(len(self.test_tweets))]

		self.train_tweets = [ElectionData.replace_target(self.train_tweets[i], self.train_targets[i]) for i in range(len(self.train_tweets))]
		self.test_tweets = [ElectionData.replace_target(self.test_tweets[i], self.test_targets[i]) for i in range(len(self.test_tweets))]

		# Padding tweets (manually adding '<PAD> tokens')
		if not self.dynamic_padding:
			self.train_tweets = util.pad_sequences(self.train_tweets, pad_location='RIGHT')
			self.test_tweets = util.pad_sequences(self.test_tweets, pad_location='RIGHT')

		# Building vocabulary
		self.vocab, self.vocab_inv = util.build_vocabulary(self.train_tweets + self.test_tweets)

		if embedding:
			# Vectorizing tweets - Glove embedding
			start = time.clock()
			print(' - Loading embedding..')
			glove, self.glove_vec, self.glove_shape, glove_vocab = util.gensim_load_vec('resources/wordemb/glove.twitter.word2vec.27B.100d.txt')
			glove_vocab = [token.encode('utf-8') for token in glove_vocab]
			self.glove_vocab_dict = {j:i for i, j in enumerate(glove_vocab)}
			self.glove_vec = np.append(self.glove_vec, [[0]*self.glove_shape[1]], axis=0)
			self.glove_shape = [self.glove_shape[0]+1, self.glove_shape[1]]
			print(' - DONE')
			print("time taken: %f mins"%((time.clock() - start)/60))

			if saved==False:
				start = time.clock()
				print(' - Matching words-indices')
				self.train_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.train_tweets])
				self.train_left_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.train_left_tweets])
				self.train_right_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.train_right_tweets])
				self.test_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.test_tweets])
				self.test_left_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.test_left_tweets])
				self.test_right_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.test_right_tweets])

				self.train_df = [(self.train_x[i], self.train_left_x[i], self.train_right_x[i], self.train_y[i]) 
								for i in range(len(self.train_x))]
				self.test_df = [(self.test_x[i], self.test_left_x[i], self.test_right_x[i], self.test_y[i]) 
								for i in range(len(self.test_x))]

				train_y = np.array([d[-1] for d in self.train_df])
				self.train_df, self.dev_df = self.build_train_dev(train_y) # Dividing to train and dev set
				print(' - DONE')
				print("time taken: %f mins"%((time.clock() - start)/60))
				print(" - Saving data")
				np.save('data/election-data/train_df.npy', self.train_df)
				np.save('data/election-data/dev_df.npy', self.dev_df)
				np.save('data/election-data/test_df.npy', self.test_df)
				print(' - DONE')
			else:
				print(" - Loading data")
				self.train_df = np.load('data/election-data/train_df.npy')
				self.dev_df = np.load('data/election-data/dev_df.npy')
				self.test_df = np.load('data/election-data/test_df.npy')
				print(' - DONE')

		else:
			# Vectorizing tweets - one-hot-vector
			self.train_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.train_tweets])
			self.test_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.test_tweets])

		self.create_batches()
		self.reset_batch_pointer()

	@staticmethod
	def read_data(data_dir):
		inputs=streamtwElec(data_dir)
		data = []
		for i in inputs:
			tw = i[0]
			target = i[1]
			if target=='"long_term_economic"_plans':
				target='long_term_economic'
			label = i[2]
			data.append([tw, target, label])
		return data

	@staticmethod
	def replace_target(tweet, target):
		tweet = list(itertools.chain.from_iterable((target.split('_')) if item == target else (item, ) for item in tweet))
		return tweet

	@staticmethod
	def split_tweet(tweet, target):
		target_index = tweet.index(target)
		left = tweet[0:target_index] + target.split('_')
		right = target.split('_') + tweet[target_index+1:]
		right = [i for i in reversed(right)]
		return left, right

	def build_train_dev(self, train_y, dev_size=0.2, random_seed=42):
		return train_test_split(
			self.train_df,
			test_size=dev_size,
			random_state=random_seed,
			stratify=train_y)

	def create_batches(self):
		self.train_df = self.shuffle_data(self.train_df) # Randomlise data
		#train set:
		self.train_x = np.array([d[0] for d in self.train_df])
		self.train_size = np.array([len(seq) for seq in self.train_x])
		self.train_y = np.array([d[-1] for d in self.train_df])
		self.train_left_x = np.array([d[1] for d in self.train_df])
		self.train_left_size = np.array([len(seq) for seq in self.train_left_x])
		self.train_right_x = np.array([d[2] for d in self.train_df])
		self.train_right_size = np.array([len(seq) for seq in self.train_right_x])
		self.train_x = util.pad_sequences(self.train_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT') # Padding
		self.train_left_x = util.pad_sequences(self.train_left_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.train_right_x = util.pad_sequences(self.train_right_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.train_x = np.array(self.train_x)
		self.train_left_x = np.array(self.train_left_x)
		self.train_right_x = np.array(self.train_right_x)
		#dev set:
		self.dev_x = np.array([d[0] for d in self.dev_df])
		self.dev_size = np.array([len(seq) for seq in self.dev_x])
		self.dev_y = np.array([d[-1] for d in self.dev_df])
		self.dev_left_x = np.array([d[1] for d in self.dev_df])
		self.dev_left_size = np.array([len(seq) for seq in self.dev_left_x])
		self.dev_right_x = np.array([d[2] for d in self.dev_df])
		self.dev_right_size = np.array([len(seq) for seq in self.dev_right_x])
		self.dev_x = util.pad_sequences(self.dev_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT') # Padding
		self.dev_left_x = util.pad_sequences(self.dev_left_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.dev_right_x = util.pad_sequences(self.dev_right_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.dev_x = np.array(self.dev_x)
		self.dev_left_x = np.array(self.dev_left_x)
		self.dev_right_x = np.array(self.dev_right_x)
		#test set:
		self.test_x = np.array([d[0] for d in self.test_df])
		self.test_size = np.array([len(seq) for seq in self.test_x])
		self.test_y = np.array([d[-1] for d in self.test_df])
		self.test_left_x = np.array([d[1] for d in self.test_df])
		self.test_left_size = np.array([len(seq) for seq in self.test_left_x])
		self.test_right_x = np.array([d[2] for d in self.test_df])
		self.test_right_size = np.array([len(seq) for seq in self.test_right_x])
		self.test_x = util.pad_sequences(self.test_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT') # Padding
		self.test_left_x = util.pad_sequences(self.test_left_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.test_right_x = util.pad_sequences(self.test_right_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.test_x = np.array(self.test_x)
		self.test_left_x = np.array(self.test_left_x)
		self.test_right_x = np.array(self.test_right_x)

		# Vectorizing labels
		self.train_y = pd.get_dummies(self.train_y).values.astype(np.int32)
		self.dev_y = pd.get_dummies(self.dev_y).values.astype(np.int32)
		self.test_y = pd.get_dummies(self.test_y).values.astype(np.int32)

		# Creating training batches
		self.num_batches = len(self.train_x)//self.batch_size
		if self.num_batches==0:
			assert False, "Not enough data for the batch size."
		self.batch_df = np.array_split(self.train_df, self.num_batches) # Splitting train set into batches based on num_batches

	def next_batch(self):
		df = self.batch_df[self.pointer]
		x = np.array([d[0] for d in df])
		xl = np.array([d[1] for d in df])
		xr = np.array([d[2] for d in df])
		y = np.array([d[-1] for d in df])
		y = pd.get_dummies(y).values.astype(np.int32)
		seq_len = [len(seq) for seq in x]
		seq_len_l = [len(seq) for seq in xl]
		seq_len_r = [len(seq) for seq in xr]
		if self.dynamic_padding:
			x = np.array(self.pad_minibatches(x, 'RIGHT'))
			xl = np.array(self.pad_minibatches(xl, 'RIGHT'))
			xr = np.array(self.pad_minibatches(xr, 'RIGHT'))
		self.pointer += 1
		return x, y, seq_len, xl, seq_len_l, xr, seq_len_r

	def reset_batch_pointer(self):
		self.train_df = self.shuffle_data(self.train_df)
		self.pointer = 0

	def pad_minibatches(self, x):
		x = util.pad_sequences(x, dynamic_padding=self.dynamic_padding, pad_location=pad_location)
		return x

	@staticmethod
	def shuffle_data(a):
		a = np.array(a)
		p = np.random.permutation(len(a))
		return a[p]