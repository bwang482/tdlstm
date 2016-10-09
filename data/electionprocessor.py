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

		# Padding tweets (manually adding '<PAD> tokens')
		if not self.dynamic_padding:
			self.train_tweets = util.pad_sequences(self.train_tweets, pad_location='RIGHT')
			self.test_tweets = util.pad_sequences(self.test_tweets, pad_location='RIGHT')

		# Vectorizing labels
		self.train_y = pd.get_dummies(self.train_y).values.astype(np.int32)
		self.test_y = pd.get_dummies(self.test_y).values.astype(np.int32)

		# Building vocabulary
		self.vocab, self.vocab_inv = util.build_vocabulary(self.train_tweets + self.test_tweets)

		if embedding:
			# Vectorizing tweets - Glove embedding
			start = time.clock()
			print(' - Loading embedding..')
			glove, self.glove_vec, self.glove_shape, glove_vocab = util.gensim_load_vec('resources/wordemb/glove.twitter.word2vec.27B.100d.txt')
			glove_vocab = [token for token in glove_vocab]
			self.glove_vocab_dict = {j:i for i, j in enumerate(glove_vocab)}
			self.glove_vec = np.append(self.glove_vec, [[0]*self.glove_shape[1]], axis=0)
			self.glove_shape = [self.glove_shape[0]+1, self.glove_shape[1]]
			print(' - DONE')
			print("time taken: %f mins"%((time.clock() - start)/60))

			if saved==False:
				start = time.clock()
				print(' - Matching words-indices')
				self.train_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.train_tweets])
				self.test_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in tweet] for tweet in self.test_tweets])
				print(' - DONE')
				print("time taken: %f mins"%((time.clock() - start)/60))
				print(" - Saving padded data")
				# self.train_x = self.train_x.astype(np.int32)
				# self.test_x = self.test_x.astype(np.int32)
				np.save('data/election-data/train_x.npy', self.train_x)
				np.save('data/election-data/test_x.npy', self.test_x)
				print(' - DONE')
			else:
				print(" - Loading padded data")
				self.train_x = np.load('data/election-data/train_x.npy')
				self.test_x = np.load('data/election-data/test_x.npy')
				print(' - DONE')

		else:
			# Vectorizing tweets - one-hot-vector
			self.train_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.train_tweets])
			self.test_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.test_tweets])

		self.train_x, self.dev_x, self.train_y, self.dev_y = self.build_train_dev(dev_size=0.2)
		self.train_size = np.array([len(seq) for seq in self.train_x])
		self.dev_size = np.array([len(seq) for seq in self.dev_x])
		self.test_size = np.array([len(seq) for seq in self.test_x])
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

	def build_train_dev(self, dev_size=0.2, random_seed=42):
		return train_test_split(
			self.train_x,
			self.train_y,
			test_size=dev_size,
			random_state=random_seed)

	def create_batches(self):
		self.num_batches = len(self.train_x)//self.batch_size
		if self.num_batches==0:
			assert False, "Not enough data for the batch size."
		train_x, train_y = self.train_x, self.train_y
		train_x, train_y, self.train_size = self.shuffle_data(self.train_x, self.train_y, self.train_size)
		self.batch_x = np.array_split(train_x, self.num_batches)
		self.batch_y = np.array_split(train_y, self.num_batches)
		self.train_x = util.pad_sequences(train_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.dev_x = util.pad_sequences(self.dev_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.test_x = util.pad_sequences(self.test_x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		self.train_x = np.array(self.train_x)
		self.dev_x = np.array(self.dev_x)
		self.test_x = np.array(self.test_x)

	def next_batch(self):
		x, y = self.batch_x[self.pointer], self.batch_y[self.pointer]
		seq_len = [len(seq) for seq in x]
		if self.dynamic_padding:
			x = np.array(self.pad_minibatches(x))
		self.pointer += 1
		return x, y, seq_len

	def reset_batch_pointer(self):
		self.train_x, self.train_y, self.train_size = self.shuffle_data(self.train_x, self.train_y, self.train_size)
		self.pointer = 0

	def pad_minibatches(self, x):
		x = util.pad_sequences(x, dynamic_padding=self.dynamic_padding, pad_location='RIGHT')
		return x

	@staticmethod
	def shuffle_data(a, b, c):
		a = np.array(a)
		b = np.array(b)
		a_size = np.array(c)
		assert len(a) == len(b) == len(c)
		p = np.random.permutation(len(a))
		return a[p], b[p], c[p]