from scipy.linalg import solve
import numpy as np

k = 8

# 常数
delta = 1e-5
# action_attacks = ['attack1', 'attack2', 'attack3', 'attack5']
action_attacks = ['attack2']
action_attack_energy_costs = [int(action_attack.strip('attack')) for action_attack in action_attacks]
actions = ['energy'] + action_attacks + ['defence']
# 只要多于一个attack，则方程组就不可解，秩少了1
# 因为此时的
num_actions = len(actions)
WIN = 1
LOSE = 0
DRAW = (WIN+LOSE) / 2

# 方程组部分
init_val = DRAW
# 保存方程组解的结果
v = init_val * np.ones((k+1, k+1)) # 全部标记为未知
# v = np.random.random((k+1, k+1))
new_v = -np.ones((k+1, k+1)) # 全部标记为unused
p = -np.ones((k+1, k+1), dtype=object) # 全部标记为unused
# 求解方程组
a = np.ones((k, k), dtype=object)

def update_v_and_p(state):
	b = np.zeros((len(a[state]), ))
	b[-1] = 1

	x = solve(a[state], b)
	# if np.linalg.matrix_rank(a[state]) < len(a[state]): # 不满秩
	# TODO: 尝试得到无限组解的基底，但是失败
	# 	_s, _v, _d = np.linalg.svd(a[state])
	# 	print(state, np.compress(_v < 1e-1, _d, axis=0), '>>>')

	# 	np.linalg.lstsq(a[state], b)

	p[state], new_v[state] = x[:-1], x[-1]
	if abs(new_v[state] - v[state]) > delta:
		v[state] = new_v[state]
		return False # not convence
	return True

# return skip_action_attacks
def upper_bound(energy):
	for action_attack_index, action_attack_energy_cost, in enumerate(action_attack_energy_costs):
		if energy < action_attack_energy_cost:
			return len(action_attack_energy_costs) - action_attack_index
	return 0
'''
类似于upper bound
e.g. 
if state[1] == 0: # 对方的气为0时，刚才对方attack1, attack2, attack3, attack5的情况直接pass，把defence覆盖上去
	a_temp[(action_b_index-4, action_a_index)] = v[(state[0]+1, state[1])]
elif state[1] < 2: # 对方的气为1时，刚才对方attack2, attack3, attack5的情况直接pass，把defence覆盖上去
	a_temp[(action_b_index-3, action_a_index)] = v[(state[0]+1, state[1])]
elif state[1] < 3: # 对方的气为2时，刚才对方attack3, attack5的情况直接pass，把defence覆盖上去
	a_temp[(action_b_index-2, action_a_index)] = v[(state[0]+1, state[1])]
elif state[1] < 5: # 对方的气为3,4时，刚才对方attack5的情况直接pass，把defence覆盖上去
	a_temp[(action_b_index-1, action_a_index)] = v[(state[0]+1, state[1])]
else:
	a_temp[action_b_a] = v[(state[0]+1, state[1])]
'''

def set_a(state):
	# 双方的气小于攻击的最低限度时，双方都应该攒气
	if state[0] < action_attack_energy_costs[0] and state[1] < action_attack_energy_costs[0]:
		new_state = (action_attack_energy_costs[0], action_attack_energy_costs[0]-state[0]+state[1])
		set_a(new_state)
		a[state] = a[new_state]
		return

	if state[1] < action_attack_energy_costs[0]: # 对方的气小于攻击的最低限度时，对方攻击、己方defence的情况直接pass
		# e.g. 没有气时，对方动作=6-4=2，只有energy和defence两行，外加最后的概率行；己方有所有列，外加-v列，除了defence
		a_temp = np.ones((2+1, num_actions+1-1))
	else: # 对方的气小于攻击的第二低的限度时，对方第二及以上攻击的情况直接pass（转置后的行数少若干），己方defence却不能pass（转置后的列数不少）
		# e.g. 一格气时，对方动作=6-3=3，有energy和attack1和defence三行，外加概率行；己方有所有列，外加-v列
		a_temp = np.ones((num_actions-upper_bound(state[1])+1, num_actions+1))
	# 如果只有一个动作，则等价于下面的写法
	# if state[1] < action_attack_energy_costs[0]: 对方有所有行，外加概率行；己方有所有列，外加-v列
	# 	a_temp = np.ones((num_actions, num_actions)) 恰好少了对方攻击、己方defence的情况，一行一列，仍然构成方阵
	# else: 对方有所有行，外加概率行；己方有所有列，外加-v列
	# 	a_temp = np.ones((num_actions+1, num_actions+1))
	# 如果有多个动作，这会导致不构成方阵，方程组无法求解

	for action_a_index, action_a in enumerate(actions):
		for action_b_index, action_b in enumerate(actions):
			# 这里的矩阵要转置
			action_b_a = (action_b_index, action_a_index)
			if action_a == 'energy':
				if action_b == 'energy': a_temp[action_b_a] = v[(state[0]+1, state[1]+1)]
				elif action_b.startswith('attack'): a_temp[action_b_a] = LOSE
				else: # defence，覆盖掉之前对方不合法的attack部分
					action_b_skip_attack_a = (action_b_index-upper_bound(state[1]), action_a_index)
					a_temp[action_b_skip_attack_a] = v[(state[0]+1, state[1])]
			elif action_a.startswith('attack'):
				energy_cost_a = int(action_a.strip('attack'))
				if action_b == 'energy': a_temp[action_b_a] = WIN
				elif action_b.startswith('attack'):
					energy_cost_b = int(action_b.strip('attack'))
					if energy_cost_a == energy_cost_b: # 均势
						a_temp[action_b_a] = v[(state[0]-energy_cost_a, state[1]-energy_cost_b)]
					elif energy_cost_a > energy_cost_b: # 己方技能更强
						a_temp[action_b_a] = WIN
					else: # 己方技能更弱
						a_temp[action_b_a] = LOSE
				else: # defence
					action_b_skip_attack_a = (action_b_index-upper_bound(state[1]), action_a_index)
					a_temp[action_b_skip_attack_a] = v[(state[0]-energy_cost_a, state[1])]
			else: # defence，覆盖掉之前己方不合法的attack部分
				action_b_a_skip_attack = (action_b_index, action_a_index-upper_bound(state[0]))
				if state[1] >= action_attack_energy_costs[0]: # 对方的气小于攻击的最低限度时，己方defence的情况直接pass
					if action_b == 'energy': a_temp[action_b_a_skip_attack] = v[(state[0], state[1]+1)]
					elif action_b.startswith('attack'):
						energy_cost_b = int(action_b.strip('attack'))
						a_temp[action_b_a_skip_attack] = v[(state[0], state[1]-energy_cost_b)]
					else: # defence，覆盖掉之前对方不合法的attack部分
						action_b_skip_attack_a_skip_attack = (action_b_index-upper_bound(state[1]), action_a_index-upper_bound(state[0]))
						a_temp[action_b_skip_attack_a_skip_attack] = v[(state[0], state[1])]

	a_temp[:, -1] = -1 # 对方各种动作的收益均等于v
	a_temp[-1, :] = 1 # 所有动作的概率之和为1
	a_temp[-1, -1] = 0
	
	a[state] = a_temp
	# e.g.
	# a[2,1] = np.array([
	# 	[v[3,2], 1, v[2,2], -1],
	# 	[0, v[1,0], v[2,0], -1],
	# 	[v[3,1], v[1,1], v[2,1], -1],
	# 	[1, 1, 1, 0],
	# ])
	# a[1,0] 划掉了一行一列
	# a[1,0] = np.array([
	# 	[v[2,1], 1, -1],
	# 	[v[2,0], 0.5, -1],
	# 	[1, 1, 0],
	# ])

def bobozan():
	# TODO：对面反对称
	v[k-1, 0] = WIN # 已知
	v[k, 1:k] = WIN # 已知
	v[k, 0] = WIN # 也已知，但是计算过程用不到，只需要最后输出结果
	v[range(k+1), range(k+1)] = DRAW

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

		for state in states: # 更新方程组
			set_a(state)
		for state in states: # 求解、判断是否收敛
			try:
				convence_flag &= update_v_and_p(state)
			except Exception as e: # 当动作个数>1时就会触发
				print(state, e, 'bad')
				print(np.linalg.matrix_rank(a[state]), len(a[state])) # a不满秩
				print(a[state])
			# else:
			# 	print(state, 'good')
			# 	print(np.linalg.matrix_rank(a[state]), len(a[state])) # a满秩
			# 	print(a[state])

		if convence_flag: break
	
	print(v[:k+1, :k+1])
	p[:action_attack_energy_costs[0], :action_attack_energy_costs[0]] = -1 # 一定是energy，故不用输出
	print(p[:k+1, :k+1])

bobozan()
