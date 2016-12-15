import numpy as np
import neiro as nn

def Softmax(x,deriv = False):
	s = np.exp(x)/np.exp(x).sum()
	if deriv:
		return s * (1 - s)
	return s

def Normal_distribution(mean, var):
	def normal(x):
		const = 1/(var*np.sqrt(2*np.pi))
		f = const * np.exp(-0.5*((x-mean)/var)**2)
		return f
	return normal


class IOHMM(object):

	def __init__(self, init_vec, num_states=3, end_up_st = 1, st_net_params = (2,3), out_net_params = (2,1), out_distr=None):
		"""
		init_vec -- одномерный массив начальных состояний
		num_states -- количество внутренних состояний
		end_up_st -- номер стостояния, начиная с нуля, в котором модель будет присутствовать в конце последовательности
		st_net_params -- параметры слоя для определения внутренних состояний
		out_net_params -- параметры слоя для определения выхода модели
		out_distr -- распределение выхода 
		
		Примем, что процесс всегда заканчивается в заданном состоянии.
		Нейросети, используемые в данномй модели всегда имеют линейные активационные функции
		"""

		self.num_states = num_states
		self.init_vec = np.array([init_vec]).T
		
		assert self.init_vec.sum() == 1
		assert len(self.init_vec) == num_states
		assert 0<=end_up_st<=(num_states-1)

		if out_distr is None:
			self.out_distr = Normal_distribution(0, 1)
		else:
			self.out_distr = out_distr

		self.states = self.init_vec.copy()
		self.end_up_st = end_up_st

		self.st_nets=[]
		self.out_nets=[]
		for i in range(num_states):
			Nls = nn.Neiro_layer(2*np.random.random(st_net_params) - 1)
			Nlo = nn.Neiro_layer(2*np.random.random(out_net_params) - 1)
			self.st_nets.append(Nls)
			self.out_nets.append(Nlo)

	def refresh(self):
		"""
		Метод для возвращения модели в исходное состояние
		"""
		self.states = self.init_vec.copy()

	def forward_step(self, inputs, y_true, ret_probs = False):
		"""
		inputs - numpy-массив вида: [u,1]

		returns вектор выхода системы в соответствии с настроёками её

		Метод реакции модели на единичный вход
		"""
		st_outs = []
		o_outs = []
		inputs = np.array(inputs)

		for n in self.st_nets:
			st_outs.append(n.forward_pass(inputs))
		for n in self.out_nets:
			o_outs.append(n.forward_pass(inputs))

		st_outs1 = np.array(list(map(Softmax, st_outs)))

		if np.isnan(st_outs1).any():
			print(st_outs)
			raise Exception('st_outs1')
		
		st_outs = np.array(st_outs1)
		o_outs = np.array(o_outs)
		out = self.states.T.dot(o_outs)
		self.states = self.states.T.dot(st_outs).T
		delt = o_outs-y_true
		if ret_probs:
			return out, st_outs, self.out_distr(delt)
		else:
			return out

	def sequence(self, inputs, y_true, ret_probs = False):
		"""
		Реакция модели на заданную последовательность
		"""
		self.refresh()
		out = []
		if ret_probs:
			st_outs = []
			o_outs = []
		for i in range(len(inputs)):
			if ret_probs:
				o, st, o_ = self.forward_step(inputs[i], y_true[i], True)
				st_outs.append(st)
				o_outs.append(self.out_distr(y_true[i]-o_))
			else:
				o = self.forward_step(inputs[i], y_true[i])
			out.append(o)	
		if ret_probs:
			return out, st_outs, o_outs
		else:
			return out

	def error(self, inputs, y_true, ret_probs = False):
		"""
		inputs - numpy-массив вида: [[u,1]]
		"""
		if ret_probs:
			out, st_outs, o_outs = self.sequence(inputs, y_true, ret_probs)
		else:
			out = self.sequence(inputs, y_true, ret_probs)

		err = np.mean(np.abs(np.array(out) - np.array(y_true)))
		if ret_probs:
			return err, st_outs, o_outs
		else:
			return err


	def fit(self, inputs, y_true, err_fit = None, max_steps=1e5):
		"""
		inputs - numpy-массив вида: [[u,1]]
		y_true - numpy-массив столбец выходных данных
		err_fit - допустимая ошибка. если параметр не указан, то обучение будет проводится max_steps эпох.
		max_steps - максимальное количество этох обучения
		"""

		inputs = np.array(inputs)
		b_init = np.zeros((1,self.num_states))
		b_init[0][self.end_up_st-1] = 1
		b_init = b_init.T
		a_init = self.init_vec.copy()

		for step in range(int(max_steps)):
			err, st_outs, o_outs = self.error(inputs, y_true, ret_probs = True)

			if np.isnan(st_outs).any():
				print(st_outs)
				raise Exception('st_outs' + str(step))

			if err_fit is not None:
				if err < err_fit:
					break

			a_lst = [a_init]
			b_lst = [b_init]
			for i in range(len(y_true)):
				a = st_outs[i].dot(a_lst[-1])*o_outs[i]
				a_lst.append(a)
				b = (o_outs[-i]*st_outs[-i].T).T.dot(b_lst[-1])
				b_lst.append(b)

			b_lst = list(reversed(b_lst))

			L = a[self.end_up_st][-1]
			g_lst=list(map(lambda x,y: x*y/L, a_lst, b_lst))
			h_lst=list(map(lambda x,y,z,f: f*((x*z).dot(y.T))/L, o_outs, a_lst[1::], b_lst, st_outs))

			g_lst=np.array(g_lst[1::])
			h_lst=np.array(h_lst)
			st_outs = np.array(st_outs)
			o_outs = np.array(o_outs)
			inp = inputs.reshape((-1,inputs.shape[1],1))

			syn_upd_st = (h_lst*(1-st_outs)) 		## delt should be summed by all times and multed to input
			syn_upd_st = syn_upd_st.reshape(syn_upd_st.shape[:-1]+(1,syn_upd_st.shape[-1]))
			
			if np.isnan(h_lst).any():
				raise Exception('syn_upd_st')
				
			syn_upd_st_ls = []
			for i in range(len(inputs)):
				syn_upd_st_ls.append(list(map(lambda st, u: u.dot(st), syn_upd_st[i], inp)))
			syn_upd_st = np.array(syn_upd_st_ls)
			syn_upd_st = syn_upd_st.sum(axis = 0)
			for n,w in zip(self.st_nets,syn_upd_st):
				n.change_weights(w, n=1)

			syn_upd_o = (g_lst*(1-o_outs)) 	## delt should be summed by all times and multed to input
			syn_upd_o = syn_upd_o.reshape(syn_upd_o.shape[:-1]+(1,syn_upd_o.shape[-1]))
			syn_upd_o_ls = []
			for i in range(len(inputs)):
				syn_upd_o_ls.append(list(map(lambda st, u: u.dot(st), syn_upd_o[i], inp)))
			syn_upd_o = np.array(syn_upd_o_ls)
			syn_upd_o = syn_upd_o.sum(axis = 0)
			for n, w in zip(self.out_nets,syn_upd_o):
				n.change_weights(w, n=1)


if __name__ == '__main__':
	init_v = np.array([0,1,0])
	inputs = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
	y_true = np.array([[0],[0.01],[0.15],[0.21],[0.34],[0.46],[0.56],[0.60],[0.67],[0.76],[0.84]])
	model = IOHMM(init_v)

	print(model.sequence(inputs,y_true))
	print(model.error(inputs,y_true))
	model.fit(inputs,y_true,max_steps=1000)
	print(model.sequence(inputs,y_true))
	print(model.error(inputs,y_true))
	'''
	b_init = np.zeros((1,model.num_states))
	b_init[0][model.end_up_st-1] = 1
	b_init = b_init.T
	a_init = model.init_vec.copy()

	err, st_outs, o_outs = model.error(inputs, y_true, ret_probs = True)
	a_lst = [a_init]
	b_lst = [b_init]
	for i in range(len(y_true)):
				a = st_outs[i].dot(a_lst[-1])*o_outs[i]
				a_lst.append(a)
				b = (o_outs[-i]*st_outs[-i].T).T.dot(b_lst[-1])
				b_lst.append(b)
	b_lst = list(reversed(b_lst))
	L = a[model.end_up_st][-1]
	g_lst=list(map(lambda x,y: x*y/L, a_lst, b_lst))
	h_lst=list(map(lambda x,y,z,f: f*((x*z).dot(y.T))/L, o_outs, a_lst[1::], b_lst, st_outs))
	g_lst=np.array(g_lst[1::])
	h_lst=np.array(h_lst)
	st_outs = np.array(st_outs)
	o_outs = np.array(o_outs)
	inp = inputs.reshape((-1,inputs.shape[1],1))
	'''