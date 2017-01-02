import itertools
import os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
from gensim import utils
from twtokenize import tokenize
import util
from sklearn.model_selection import train_test_split


class streamtw(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.i=0
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):

            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                self.i+=1
                if self.i==1:
                    tw=line.lower().strip()
                if self.i==2:
                    target=line.lower().strip()
                if self.i==3:
                    senti=int(line.strip())+2
                    tw=tw.replace(target,' '+target+' ')
                    tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
                    tw=tw.replace(target,' '+'_'.join(target.split())+' ')
                    tweet=tokenize(tw)
                    yield (tweet,'_'.join(target.split()),senti)
                    self.i=0


class LidongData:
	def __init__(self, batch_size, dynamic_padding=False, preprocessing=False, embedding=True, saved=False, max_length=None):
		train = LidongData.read_data('../data/lidong-data/training/')
		test = LidongData.read_data('../data/lidong-data/testing/')
		self.batch_size = batch_size
		self.dynamic_padding = dynamic_padding
		self.train_tweets, self.train_targets, self.train_y = zip(*train)
		self.test_tweets, self.test_targets, self.test_y = zip(*test)

		self.train_left_tweets = [LidongData.split_tweet(self.train_tweets[i], self.train_targets[i])[0] for i in range(len(self.train_tweets))]
		self.train_right_tweets = [LidongData.split_tweet(self.train_tweets[i], self.train_targets[i])[1] for i in range(len(self.train_tweets))]
		self.test_left_tweets = [LidongData.split_tweet(self.test_tweets[i], self.test_targets[i])[0] for i in range(len(self.test_tweets))]
		self.test_right_tweets = [LidongData.split_tweet(self.test_tweets[i], self.test_targets[i])[1] for i in range(len(self.test_tweets))]

		self.train_tweets = [LidongData.replace_target(self.train_tweets[i], self.train_targets[i]) for i in range(len(self.train_tweets))]
		self.test_tweets = [LidongData.replace_target(self.test_tweets[i], self.test_targets[i]) for i in range(len(self.test_tweets))]
		self.train_targets = [train_target.split('_') for train_target in self.train_targets]
		self.test_targets = [test_target.split('_') for test_target in self.test_targets]

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
			glove, self.glove_vec, self.glove_shape, glove_vocab = util.gensim_load_vec('../resources/wordemb/glove.twitter.word2vec.27B.100d.txt')
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
				self.train_target_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in target] for target in self.train_targets])
				self.test_target_x = np.array([[self.glove_vocab_dict[token] if token in glove_vocab else 1193514 for token in target] for target in self.test_targets])

				self.train_df = [(self.train_x[i], self.train_left_x[i], self.train_right_x[i], self.train_target_x[i], self.train_y[i]) 
								for i in range(len(self.train_x))]
				self.test_df = [(self.test_x[i], self.test_left_x[i], self.test_right_x[i], self.test_target_x[i], self.test_y[i]) 
								for i in range(len(self.test_x))]

				train_y = np.array([d[-1] for d in self.train_df])
				self.train_df, self.dev_df = self.build_train_dev(train_y) # Dividing to train and dev set
				print(' - DONE')
				print("time taken: %f mins"%((time.clock() - start)/60))
				print(" - Saving data")
				np.save('../data/lidong-data/train_df.npy', self.train_df)
				np.save('../data/lidong-data/dev_df.npy', self.dev_df)
				np.save('../data/lidong-data/test_df.npy', self.test_df)
				print(' - DONE')
			else:
				print(" - Loading data")
				self.train_df = np.load('../data/lidong-data/train_df.npy')
				self.dev_df = np.load('../data/lidong-data/dev_df.npy')
				self.test_df = np.load('../data/lidong-data/test_df.npy')
				print(' - DONE')

		else:
			# Vectorizing tweets - one-hot-vector
			self.train_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.train_tweets])
			self.test_x = np.array([[self.vocab[token] for token in tweet] for tweet in self.test_tweets])

		self.create_batches()
		self.reset_batch_pointer()

	@staticmethod
	def read_data(data_dir):
		twtriple=streamtw(data_dir)
		data = []
		for triple in twtriple:
			tw = triple[0]
			target = triple[1]
			label = triple[2]
			# loc = tw.index(target)
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
		self.train_target_x = np.array([d[3] for d in self.train_df])
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
		self.dev_target_x = np.array([d[3] for d in self.dev_df])
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
		self.test_target_x = np.array([d[3] for d in self.test_df])
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
		tar = np.array([d[3] for d in df])
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
		return x, y, seq_len, xl, seq_len_l, xr, seq_len_r, tar

	def reset_batch_pointer(self):
		self.train_df = self.shuffle_data(self.train_df)
		self.pointer = 0

	def pad_minibatches(self, x, pad_location):
		x = util.pad_sequences(x, dynamic_padding=self.dynamic_padding, pad_location=pad_location)
		return x

	@staticmethod
	def shuffle_data(a):
		a = np.array(a)
		p = np.random.permutation(len(a))
		return a[p]


# Below use some Tensorflow inbuilt i/o and batching functions:

def make_seq_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token in sequence:
        fl_tokens.feature.add().int64_list.value.append(token)
    for label in labels:
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _,examples = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=examples,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return context_parsed, sequence_parsed


def write_tfrecord(data, data_path):
	x = data.train_x.tolist()
	y = data.train_y.astype(int).tolist()
	with open(data_path, 'a') as fp:
	    writer = tf.python_io.TFRecordWriter(fp.name)
	    for sequence, label_sequence in zip(x, y):
	        ex = make_seq_example(sequence, label_sequence)
	        writer.write(ex.SerializeToString())
	    writer.close()
	    print("Wrote to {}".format(fp.name))


def tf_batch(x, y, batch_size):
	with tf.device("/cpu:0"):
		x = tf.convert_to_tensor(x, dtype=tf.int32)
		y = tf.convert_to_tensor(y, dtype=tf.int32)
		batched_x, batched_y = tf.train.batch(
			tensors=[x, y],
			batch_size=batch_size,
			dynamic_pad=True,
			enqueue_many=True,
			name='batching'
			)
	return (batched_x, batched_y)