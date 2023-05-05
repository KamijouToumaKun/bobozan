from scipy.optimize import linprog
import numpy as np
k = 8

# 常数
delta = 1e-5
action_attacks = ['attack1', 'attack2', 'attack3']
actions = ['energy'] + action_attacks + ['breakthrough5'] + ['defence']
'''
加上破防之后，概率部分收敛到的结果更不唯一了
例如<6,0>也是必胜的，但获胜途径有多条
可以直接攻击破防，如果还没死的话就再攻击一次而胜利
也可以吸收变成<7,1>，然后再按照上述方案走：同样是必胜之势，只是对方可能多拖一轮
本文件算出来，吸气和破防的概率一般是五五开的，如：[0.53 0.   0.   0.   0.47 0.  ]
'''
action_attack_energy_costs = [int(action_attack.strip('attack')) for action_attack in action_attacks]
num_actions = len(actions)
WIN = 1
LOSE = 0
DRAW = (WIN+LOSE) / 2

# 方程组部分
init_val = DRAW
# 保存方程组解的结果
# v = init_val * np.ones((2, 2, k+1, k+1)) # 全部标记为未知
v = np.random.random((2, 2, k+1, k+1)) # TODO: 这样写的随机性更高，但每次输出结果都不同
new_v = -np.ones((2, 2, k+1, k+1)) # 全部标记为unused
p = -np.ones((2, 2, k+1, k+1), dtype=object) # 全部标记为unused
# 求解方程组
a = np.ones((2, 2, k, k), dtype=object)

def update_v_and_p(state, no_defence):
	# 外加最后的概率行；外加-v列
	c = np.zeros(num_actions+1)
	c[0] = -1
	A_ub = np.ones((num_actions, num_actions+1))
	# A_ub[:, 0] = 1
	A_ub[:, 1:] = -a[no_defence][state]
	b_ub = np.zeros(num_actions)
	A_eq = np.ones((1, num_actions+1))
	A_eq[0, 0] = 0
	b_eq = [1] # 所有动作的概率之和为1
	bounds = ((None, None),) + ((0, 1),) * num_actions # v的取值范围任意；p的取值范围为[0,1]
	'''
	e.g. min -V + 0p1 + 0p2 + 0p3
    V -第1列 点乘 p1,p2,p3 <= 0
    V -第2列 点乘 p1,p2,p3 <= 0
    V -第3列 点乘 p1,p2,p3 <= 0
    0V + 1p1 + 1p2 + 1p3= 1
    0 <= p1 <= 1
    0 <= p2 <= 1
    0 <= p3 <= 1
    '''
	
	res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
	if res.success:
		new_v[no_defence][state], p[no_defence][state] = res.x[0], res.x[1:]
		if abs(new_v[no_defence][state] - v[no_defence][state]) > delta:
			v[no_defence][state] = new_v[no_defence][state]
			return False # not convence
		return True
	else:
		raise ValueError(res.message)

def set_a(state, no_defence):
	set_a_lose(state, no_defence=no_defence) # 遇到犯规攻击的行为，直接判输
	# set_a_defence(state) # 遇到犯规攻击的行为，判定为defence，而攻击无效
	# 当存在破防时，破防后仍然防守是犯规的，所以不能这样
	# set_a_energy(state) 不过倒是可以判定为聚气

def set_a_lose(state, no_defence):
	a_temp = np.ones((num_actions, num_actions))
	if state[0] < state[1]:
		if no_defence[0] == no_defence[1]: # 双方破防或者都不破防
			return # 直接不用计算，答案就是1/2
		else:
			new_state = (state[1], state[0])
			new_no_defence = 1-no_defence
			set_a_lose(new_state, no_defence=new_no_defence) # 取对称的情况
			a_temp = a[new_no_defence][new_state]
			return

	for action_a_index, action_a in enumerate(actions):
		for action_b_index, action_b in enumerate(actions):
			# 这里的矩阵要转置
			action_b_a = (action_b_index, action_a_index)
			if action_a == 'energy':
				if action_b == 'energy': a_temp[action_b_a] = v[no_defence][(state[0]+1, state[1]+1)]
				elif action_b.startswith('attack') or action_b.startswith('breakthrough'):
					energy_cost_b = int(action_b.strip('attack').strip('breakthrough'))
					if state[1] < energy_cost_b: # 对方攻击犯规
						a_temp[action_b_a] = WIN
					else:
						a_temp[action_b_a] = LOSE
				else: # defence
					if no_defence[1] == 1: # 对方防御犯规
						a_temp[action_b_a] = WIN
					else:
						a_temp[action_b_a] = v[no_defence][(state[0]+1, state[1])]
			elif action_a.startswith('attack') or action_a.startswith('breakthrough'):
				energy_cost_a = int(action_a.strip('attack').strip('breakthrough'))
				if state[0] < energy_cost_a: # 己方攻击犯规
					if action_b.startswith('attack') or action_b.startswith('breakthrough'):
						energy_cost_b = int(action_b.strip('attack').strip('breakthrough'))
						if state[1] < energy_cost_b: # 对方攻击犯规
							a_temp[action_b_a] = DRAW # 双方攻击犯规
						else:
							a_temp[action_b_a] = LOSE
					else:
						a_temp[action_b_a] = LOSE
				else:
					if action_b == 'energy': a_temp[action_b_a] = WIN
					elif action_b.startswith('attack') or action_b.startswith('breakthrough'):
						energy_cost_b = int(action_b.strip('attack').strip('breakthrough'))
						if state[1] < energy_cost_b: # 对方攻击犯规
							a_temp[action_b_a] = WIN
						else:
							if energy_cost_a == energy_cost_b: # 均势
								a_temp[action_b_a] = v[no_defence][(state[0]-energy_cost_a, state[1]-energy_cost_b)]
							elif energy_cost_a > energy_cost_b: # 己方技能更强
								a_temp[action_b_a] = WIN
							else: # 己方技能更弱
								a_temp[action_b_a] = LOSE
					else: # defence
						if no_defence[1] == 1: # 对方防御犯规
							a_temp[action_b_a] = WIN
						else:
							if action_a.startswith('breakthrough'): # b被破防
								a_temp[action_b_a] = v[(no_defence[0], 1)][(state[0]-energy_cost_a, state[1])] # b被破防
							else:
								a_temp[action_b_a] = v[no_defence][(state[0]-energy_cost_a, state[1])]
			else: # defence
				if no_defence[0] == 1: # 己方防御犯规
					if action_b == 'defence' and no_defence[1] == 1: # 对方防御犯规
						a_temp[action_b_a] = DRAW # 双方防御犯规
					else:
						a_temp[action_b_a] = LOSE
				else:
					if action_b == 'energy': a_temp[action_b_a] = v[no_defence][(state[0], state[1]+1)]
					elif action_b.startswith('attack') or action_b.startswith('breakthrough'):
						energy_cost_b = int(action_b.strip('attack').strip('breakthrough'))
						if state[1] < energy_cost_b: # 对方攻击犯规
							a_temp[action_b_a] = WIN
						else:
							if action_b.startswith('breakthrough'):
								a_temp[action_b_a] = v[(1, no_defence[1])][(state[0], state[1]-energy_cost_b)] # a被破防
							else:
								a_temp[action_b_a] = v[no_defence][(state[0], state[1]-energy_cost_b)]
					else: # defence
						if no_defence[1] == 1: # 对方防御犯规
							a_temp[action_b_a] = WIN
						else:
							a_temp[action_b_a] = v[no_defence][state]

	
	a[no_defence][state] = a_temp
	# e.g.
	# a[2,1] = np.array([
	# 	[v[3,2], 1, v[2,2]],
	# 	[0, v[1,0], v[2,0]],
	# 	[v[3,1], v[1,1], v[2,1]]

def bobozan():
	# TODO：对面反对称
	v[:, :, k-1, 0] = WIN # 已知
	v[:, :, k, 1:k] = WIN # 已知
	v[:, :, k, 0] = WIN # 也已知，但是计算过程用不到，只需要最后输出结果
	# v[range(2), range(2)][range(k+1), range(k+1)] = DRAW # 双方都有防或者破防时，才是对称的
	for no_defence in [(0,0),(1,1)]:
		v[no_defence][range(k+1), range(k+1)] = DRAW

	states = []
	for m in range(3, k+1): # m = 3,4,...,k
		for n in range(m-2, 0, -1):
			states.append((m-1, n)) # (m-1, m-2); (m-1, m-3); ...; (m-1, 1)
		states.append((m-2, 0))
	# print(states)
	# e.g.
	# if k >= 3: # 带来自己的21、10
	# 	states.extend([(2,1), (1,0)])
	# if k >= 4: # 带来自己的32、31、20；不用刷新21
	# 	states.extend([(3,2), (3,1), (2,0)])
	# if k >= 5: # 带来自己的43、42、41、30；不用刷新32、31
	# 	states.extend([(4,3), (4,2), (4,1), (3,0)])
	# ...

	while True:
		convence_flag = True

		for no_defence in [(0,0),(0,1),(1,0),(1,1)]:
			for state in states: # 更新方程组
				set_a(state, no_defence=no_defence)
		for no_defence in [(0,0),(0,1),(1,0),(1,1)]:
			for state in states: # 求解、判断是否收敛
				try:
					convence_flag &= update_v_and_p(state, no_defence=no_defence)
				except Exception as e:
					print(state, e, 'bad')
					print(np.linalg.matrix_rank(a[no_defence][state]), len(a[no_defence][state])) # a不满秩
					print(a[no_defence][state])
				# else:
				# 	print(state, 'good')
				# 	# print(np.linalg.matrix_rank(a[no_defence][state]), len(a[no_defence][state])) # a满秩
				# 	print(a[no_defence][state])

		if convence_flag: break
	
	for no_defence in [(0,0),(0,1),(1,0),(1,1)]:
		print(v[no_defence][:k+1, :k+1])
		# print(p[:k+1, :k+1])
		for i in range(k+1):
			for j in range(k+1):
				print(np.around(p[no_defence][i,j], decimals=2), end=',')
			print('\n')

bobozan()
