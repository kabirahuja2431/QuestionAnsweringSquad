from qa_model import *

class MCQModel(QAModel):
	def __init__(self, FLAGS, id2word, word2id, vocab_size):
		"""
		Initializes the QA model.

		Inputs:
		  FLAGS: the flags passed in from main.py
		  id2word: dictionary mapping word idx (int) to word (string)
		  word2id: dictionary mapping word (string) to word idx (int)
		  emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
		"""
		print "Initializing the QAModel..."
		self.FLAGS = FLAGS
		self.id2word = id2word
		self.word2id = word2id

		# Add all parts of the graph
		with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
			self.add_placeholders()
			self.add_embedding_layer(vocab_size)
			self.build_graph()
			self.add_loss()

		# Define trainable parameters, gradient, gradient norm, and clip by gradient norm
		params = tf.trainable_variables()
		gradients = tf.gradients(self.loss, params)
		self.gradient_norm = tf.global_norm(gradients)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
		self.param_norm = tf.global_norm(params)

		# Define optimizer and updates
		# (updates is what you need to fetch in session.run to do a gradient update)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
		self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

		# Define savers (for checkpointing) and summaries (for tensorboard)
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
		self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
		self.summaries = tf.summary.merge_all()


	def add_placeholders(self):
		"""
		Add placeholders to the graph. Placeholders are used to feed in inputs.
		"""
		# Add placeholders for inputs.
		# These are all batch-first: the None corresponds to batch_size and
		# allows you to run the same model with variable batch_size
		self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
		self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
		self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
		self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
		self.options_ids = tf.placeholder(tf.int32, shape = [None, 4])
		self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

		# Add a placeholder to feed in the keep probability (for dropout).
		# This is necessary so that we can instruct the model to use dropout when training, but not when testing
		self.keep_prob = tf.placeholder_with_default(1.0, shape=())

	def add_embedding_layer(self, vocab_size):
		"""
		Adds word embedding layer to the graph.

		Inputs:
		  emb_matrix: shape (400002, embedding_size).
			The GloVe vectors, plus vectors for PAD and UNK.
		"""
		with vs.variable_scope("embeddings"):

            #Randomly initializing the embedding matrix
			embedding_matrix = tf.get_variable("embedding_matrix", shape = [vocab_size, self.FLAGS.embedding_size], initializer =  tf.contrib.layers.xavier_initializer())

			# Get the word embeddings for the context and question and options,
			# using the placeholders self.context_ids, self.qn_ids and self.options_ids
			self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
			self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)
            self.opt_embs = embedding_ops.embedding_lookup(embedding_matrix, self.options_ids)

	def build_graph(self):
		"""Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

		Defines:
		  self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
			These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
			Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
		  self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
			These are the result of taking (masked) softmax of logits_start and logits_end.
		"""

		# Use a RNN to get hidden states for the context and the question
		# Note: here the RNNEncoder is shared (i.e. the weights are the same)
		# between the context and the question.
		encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
		context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
		question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

		# Use context hidden states to attend to question hidden states
		attn_layer = ScaledDotAttention(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
		_, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

		# Concat attn_output to context_hiddens to get blended_reps
		blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        #context to options attention
        opt_attn_layer = ScaledDotAttention(self.keep_prob, self.FLAGS.hidden_size, self.FLAGS.hidden_size)
        _,opt_attn_output = opt_attn_layer.build_graph(blended_reps, context_mask, self.opt_embs)

		blended_reps_final = tf.contrib.layers.fully_connected(opt_attn_output, num_outputs = self.FLAGS.hidden_size)

		# Use softmax layer to compute probability distribution for the correct answer
		# Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
		with vs.variable_scope("output"):
			softmax_layer_start = SoftmaxLayer()
			self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final)
