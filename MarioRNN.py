import tensorflow as tf
import math

class MarioRNN(object):
	def __init__(self, data, rnn_sizes, max_grad_norm, dropout_keep, variational_recurrent, train, loss_function):
		self.data = data
		self.dropout_keep = dropout_keep
		self.variational_recurrent = variational_recurrent
		self.loss_function = loss_function
		self.rnn_sizes = rnn_sizes
		init_scale = 0.5

		with tf.variable_scope("RNN"):
			self.layers = [tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple=True) for size in rnn_sizes]

		self.softmax_w = tf.Variable(tf.truncated_normal([rnn_sizes[-1], data.output_size], stddev=1.0 / math.sqrt(float(data.input_size))))
		self.softmax_b = tf.Variable(tf.zeros([data.output_size]))
		self.max_grad_norm = max_grad_norm

		(self.initial_state, self.final_state, _, _, self.train_op) = self.build_graph(
			train=True,
			validate=False,
		)

		(_, _, _, self.cost, _) = self.build_graph(
			train=False,
			validate=True,
		)

		(self.single_initial_state, self.single_state, self.single_prediction, _, _) = self.build_graph(
			train=False,
			validate=False,
		)


	def build_graph(self, train, validate):
		data = self.data
		rnn_sizes = self.rnn_sizes
		max_grad_norm = self.max_grad_norm
		cells = self.layers
		if train:
			batch_size = data.batch_size
			cells = [tf.contrib.rnn.DropoutWrapper(
				cell,
				output_keep_prob=self.dropout_keep,
				variational_recurrent=self.variational_recurrent,
				dtype=data.dtype) for cell in cells]
		elif validate:
			batch_size = data.batch_size
		else:
			batch_size = 1
		cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple = True)
		initial_state = cell.zero_state(batch_size, dtype=data.dtype)
		outputs = []
		if train or validate:
			with tf.variable_scope("RNN"):
				for time_step in range(data.num_steps):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(data.input[time_step, :, :], initial_state)
					outputs.append(cell_output)
					final_state = state
			logits = [tf.matmul(lstm_output, self.softmax_w) + self.softmax_b for lstm_output in outputs]
			predictions = tf.sigmoid(logits)
			cost = tf.losses.sigmoid_cross_entropy(data.output,logits,weights=data.cost_weight)
			if train:
				tvars = tf.trainable_variables()
				grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
				#global_step = tf.Variable(0, trainable=False)
				#starter_learning_rate=0.1
				#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
				#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step) 
				optimizer = tf.train.AdamOptimizer()
				train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
			if validate:
				train_op = None	
		else:
			self.single_input = tf.placeholder(shape=[1,data.input_size], dtype=data.dtype, name='single_input')
			initial_state = state = cell.zero_state(1, dtype=data.dtype)
			(cell_output, final_state) = cell(self.single_input, state)
			predictions = tf.sigmoid(tf.matmul(cell_output,self.softmax_w) + self.softmax_b, name='single_prediction')
			cost = None
			train_op = None
		return initial_state, final_state, predictions, cost, train_op
