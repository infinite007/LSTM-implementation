import numpy as np

class lstm:
	def __init__(self, size):
		self.h = np.random.rand(size, size)
		self.input_bias = np.zeros((1, size))
		self.forget_bias = np.zeros((1, size))
		self.output_bias = np.zeros((1, size))
		self.cell_state_bias = np.zeros((1, size))
		self.input_W = np.random.rand(2 * size, size)
		self.forget_W = np.random.rand(2 * size, size)
		self.output_W = np.random.rand(2 * size, size)
		self.cell_state_W = np.random.rand(2 * size, size)
		self.cell_state = np.random.rand(size, size)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-1 * x))

	def forget_op(self):
		fo = np.matmul(self.concatenated, self.forget_W)
		fo_bias = fo + self.forget_bias
		return self.sigmoid(fo_bias)

	def input_op(self):
		inp = np.matmul(self.concatenated, self.input_W)
		inp_bias = inp + self.input_bias
		return self.sigmoid(inp_bias)

	def output_op(self):
		out = np.matmul(self.concatenated, self.output_W)
		out_bias = out + self.output_bias
		return self.sigmoid(out_bias)

	def input_h_concat(self, x):
		self.concatenated = np.concatenate((self.h, x), axis=1)

	def compute_current_state(self):
		c_prime_t_matmul = np.matmul(self.concatenated, self.cell_state_W)
		c_prime_t_bias = np.add(c_prime_t_matmul, self.cell_state_bias)
		c_prime_t_tanh = np.tanh(c_prime_t_bias)
		c_prime_t = c_prime_t_tanh
		c_t = np.multiply(self.forget_op(), self.cell_state) + np.multiply(self.input_op(), c_prime_t)
		self.cell_state = c_t
		return c_t

	def compute_output(self, x):
		self.input_h_concat(x)
		o_t = self.output_op()
		c_t = self.compute_current_state()
		h_t = np.multiply(o_t, np.tanh(c_t))
		self.h = h_t
		return h_t, c_t