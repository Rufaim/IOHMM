reshapeimport numpy as np

'''
Некоторые активационные функции и их производные
'''
def linear(x, deriv = False):
	if deriv:
		return np.ones(np.array(x).shape)
	return x

def sigmoid(x, deriv = False):
	s = 1 / (1 + np.exp(-x))
	if deriv:
		return s * (1 - s)
	return s


class Neiron(object):
	"""
	Нейрон с произвольной активационной функцией и возможность однократного обновления весов по методу градиентного спуска.
	"""
	def __init__(self, weights, activation_func=None):
		"""
		Инициализируем нейрон.
		weights - numpy-вектор весов нейрона формы (m,), weights[m-1] - смещение
		"""
		assert type(weights) is np.ndarray

		if (activation_func is None):
			self.activation_func = linear
		else:
			self.activation_func = activation_func
		self.synapses = np.array(weights)

	def forward_pass(self, inputs):
		"""
		single_input - numpy-вектор входов формы (m,), с учётом смещения
		single_input[m-1] - единица
		"""
		assert type(inputs) is np.ndarray

		return self.activation_func(inputs.dot(self.synapses))

	def update_weights(self, val_err, val_pred, val_inp, n=1):
		"""
		n - коэффициент скорости обучения
		"""
		delt = val_err*self.activation_func(val_pred, True)
		self.synapses += n*val_inp.T.dot(delt)

	def error(self, inputs, val_true):
		assert inputs.shape[1] == self.num_inputs()
		
		val_pred = self.forward_pass(inputs)
		return np.mean(np.abs(val_true - val_pred))

	def change_weights(self, synapses_upd, n=1):
		assert synapses_upd.shape == self.synapses.shape
		self.synapses += n*synapses_upd

	def num_inputs(self):
		return self.synapses.shape[0]

class Neiro_layer(Neiron):
	"""
	Однослойная нейросеть с произвольной активационной функцией, одинаковой для всех нейронов,
	и возможность однократного обновления весов по методу градиентного спуска.
	"""
	def __init__(self, weights, activation_func=None):
		"""
		weights - матрица весов нейронов формы (n,m), weights[][m-1] - смещения
		"""
		Neiron.__init__(self,weights, activation_func)

	def update_weights(self, val_err, val_pred, val_inp, n=1):
		"""
		n - коэффициент скорости обучения
		"""
		delt = val_err*self.activation_func(val_pred, True)
		last_errs = delt.dot(self.synapses.T)
		self.synapses += n*val_inp.T.dot(delt)
		return last_errs

	def num_outputs(self):
		return self.synapses.shape[1]


class Neiro_Net(object):
	"""
	Нейросеть с произвольным числом слоёв. Каждый слой имеет общую активационную функцию.
	Структура сети задаётся списком элементов вида [num_neirons, weights, activation_func], 
	где num_neirons - количество нейронов в слое, weights - numpy-матрица весов слоя,
	activation_func - активационная функция слоя вида def f(x,deriv=False), которая возвращает собственную производную при deriv=True.
	"""
	def __init__(self, num_inp=2, struct=[[1,None,None]], n=1):
		"""
		struct - список параметров для слоёв [[num_neirons,weights, activation_func]].
		Веса задаются в виде numpy-матриц (n_inp,m_neirons)
		Если веса не указаны, то они будут назначены случайным образом. 
		Если не указана функция активации, то она будет выбрана по умолчанию.
		В данных учитывается смещение нейрона.
		"""
		self.layers = []
		self.learn_rate = n
		inps = num_inp
		for p in struct:
			self.add_layer([inps]+p)
			inps = p[0]

	def add_layer(self, params=[2,1,None,None]):
		assert len(params) == 4

		inp,n,weights,activation_func = params
		if weights is None:
			#инициализация весов случайным образом со средним 0
			weights = 2*np.random.random((inp,n)) - 1 
		Nl=Neiro_layer(weights,activation_func)
		self.layers.append(Nl)

	def forward_pass(self, inputs, sub_outs=False):
		"""
		inputs - numpy-матрица (n,m), n - число примеров, m - вектор входов для каждого примера
		sub_outs - сохранение выходов промежуточных слоёв
		"""
		assert inputs.shape[1] == self.layers[0].num_inputs()

		so = [self.layers[0].forward_pass(inputs)]
		for Nlayer in self.layers[1:]:
			so.append(Nlayer.forward_pass(so[-1]))
		if sub_outs:
			return so
		else:
			return so[-1]

	def error(self, inputs, val_true, ret_preds=False):
		"""
		inputs - numpy-матрица входных данных
		val_true - numpy-вектор столбец выходных данных
		ret_preds - ключ, будут ли возвращены также массив значений с каждого слоя
		"""
		assert inputs.shape[1] == self.layers[0].num_inputs()
		
		if ret_preds:
			so = self.forward_pass(inputs, sub_outs=True)
			return (val_true - so[-1]), so
		else:
			val_pred = self.forward_pass(inputs)
			return np.mean(np.abs(val_true - val_pred))

	def fit_net(self, inputs, y_true, err_fit = None, max_steps=1e6):
		"""
		inputs - numpy-матрица входных данных
		y_true - numpy-вектор столбец выходных данных
		err_fit - допустимая ошибка. если параметр не указан, то обучение будет проводится max_steps эпох.
		max_steps - максимальное количество этох обучения
		"""
		assert 1 <= len(inputs.shape) <=2		
		assert inputs.shape[-1] == self.layers[0].num_inputs()

		for step in range(int(max_steps)):
			err , sub_outputs = self.error(inputs, y_true, ret_preds= True)

			if err_fit is not None:
				if err < err_fit:
					break
			if len(self.layers) > 1:
				errs = self.layers[-1].update_weights( err, sub_outputs[-1], sub_outputs[-2], n=self.learn_rate)
				for i in range(len(sub_outputs)-2,1,-1):
					errs=self.layers[i].update_weights(errs, sub_outputs[i], sub_outputs[i-1], n=self.learn_rate)
				self.layers[0].update_weights(errs, sub_outputs[0], inputs, n=self.learn_rate)
			else:
				self.layers[0].update_weights(err, sub_outputs[0], inputs, n=self.learn_rate)



if __name__ == '__main__':
	X = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1]])
	y = np.array([[0],
				[1],
				[1],
				[0]])

	NN = Neiro_Net(3,[[4,None,sigmoid],[1,None,sigmoid]])
	NN.fit_net(X,y,max_steps=100000)
	print("Fited net: \n" + str(NN.forward_pass(X)))
	print("Err: " + str(NN.error(X,y)))