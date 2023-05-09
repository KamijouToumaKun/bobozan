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
# action_attack_energy_costs = [int(action_attack.strip('attack')) for action_attack in action_attacks]
flag_min_action_attack_energy_cost = False
min_action_attack_energy_cost = int(action_attacks[0].strip('attack'))
# 此外，有时候解出来的概率同是最优的，但是看起来是在“浪”
# 例如，当对方的气不够最低攻击限度时，己方不应该选择无谓的防御，即p(defence)应该为0，在 solve.py 中就是这么做的
# “浪”的选择中，可能会选择一两次无谓的防御，只是最后仍能获得胜利
# 为了简化局势，这里也提供了 flag_min_action_attack_energy_cost 这一选项，默认是关闭的
# 如果打开，则求解时加入人为限制，不考虑defence这一选项，p中直接少了这一维度

num_actions = len(actions)
WIN = 1
LOSE = 0
DRAW = (WIN+LOSE) / 2
UNUSED = -1

# 方程组部分
INIT_VAL = DRAW
# 保存方程组解的结果
# v = INIT_VAL * np.ones((2, 2, k+1, k+1)) # 全部标记为未知
v = np.random.random((2, 2, k+1, k+1)) # 这样写的随机性更高，但每次输出结果都不同
new_v = UNUSED * np.ones((2, 2, k+1, k+1)) # 全部标记为unused
p = UNUSED * np.ones((2, 2, k+1, k+1), dtype=object) # 全部标记为unused
# 求解方程组
a = np.ones((2, 2, k, k), dtype=object)

def update_v_and_p(state, no_defence):
	# 可以少一列：对方不能攻击时，己方不用做无谓的defence
	if flag_min_action_attack_energy_cost and state[1] < min_action_attack_energy_cost:
		c = np.zeros(num_actions)
		A_ub = np.ones((num_actions, num_actions))
		A_eq = np.ones((1, num_actions))
		bounds = ((None, None),) + ((0, 1),) * (num_actions-1)
	else:
		# 外加最后的概率行；外加-v列
		c = np.zeros(num_actions+1)
		A_ub = np.ones((num_actions, num_actions+1))
		A_eq = np.ones((1, num_actions+1))
		bounds = ((None, None),) + ((0, 1),) * num_actions # v的取值范围任意；p的取值范围为[0,1]
	
	c[0] = -1
	# A_ub[:, 0] = 1
	A_ub[:, 1:] = -a[no_defence][state]
	b_ub = np.zeros(num_actions)
	A_eq[0, 0] = 0
	b_eq = [1] # 所有动作的概率之和为1
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
	# solve.py里还会少掉违规攻击的行和列，但用线性规划时不用排除这些违规情况
	if flag_min_action_attack_energy_cost and state[1] < min_action_attack_energy_cost: # 可以少一列：对方不能攻击时，己方不用做无谓的defence
		a_temp = np.ones((num_actions, num_actions-1)) # 不然，计算出的 p(defence) 真的可能大于0
	else:
		a_temp = np.ones((num_actions, num_actions))

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
								a_temp[action_b_a] = v[(no_defence[0], 1)][(state[0]-energy_cost_a, state[1])]
							else:
								a_temp[action_b_a] = v[no_defence][(state[0]-energy_cost_a, state[1])]
			else: # defence
				# 可以少一列：对方不能攻击时，己方不用做无谓的defence
				if flag_min_action_attack_energy_cost and state[1] < min_action_attack_energy_cost:
					continue

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

def bobozan(no_defence):
	#############################
	# 方案2.当求解的state为整个下三角时
	#############################
	states = []
	for m in range(1, k): # m = 1,2,3,4,...,k-1
		for n in range(m+1): # n = 0,1,2,3,...,m, n<=m
			states.append((m, n)) # (m-1, m-2); (m-1, m-3); ...; (m-1, 1)
	
	# if no_defence[0] == no_defence[1]:
	# 	p[no_defence][np.triu_indices_from(p[(0,0)], k=1)] = UNUSED # TODO: UNUSED = LOSE 这样好吗
	# 	v[no_defence][np.triu_indices_from(v[(0,0)], k=1)] = UNUSED
	
	# 共有部分：下三角部分置为WIN
	v[no_defence][np.tril_indices_from(v[(0,0)], k=-(k-min_action_attack_energy_cost))] = WIN
	# TODO: 把这部分的p标记为努力攒气 p(energy)=1，而不考虑其他动作
	
	# 开始求解
	while True:
		convence_flag = True

		for state in states: # 更新方程组
			if no_defence[0] == no_defence[1] and state[0] == state[1]:
				continue # 双方都破防或者不破防时，state对称的情况不用求解，根据对称性，答案就是DRAW
			# 双方防御状态不对称时，state对称的情况也要求解
			set_a(state, no_defence=no_defence)
		for state in states: # 求解、判断是否收敛
			if no_defence[0] == no_defence[1] and state[0] == state[1]:
				continue # 双方都破防或者不破防时，state对称的情况不用求解，根据对称性，答案就是DRAW
			# 双方防御状态不对称时，state对称的情况也要求解
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
	
def print_v_and_p(no_defence):
	print('==========================', no_defence, '==========================')
	print(v[no_defence][:k+1, :k+1])
	# TODO: 以后再解决p的问题：p可能是多解
	# print(p[no_defence][:k+1, :k+1])
	# for i in range(k+1):
	# 	for j in range(k+1):
	# 		print(np.around(p[no_defence][i,j], decimals=2), end=',')
	# 	print('\n')

def main():
	# 由于破防不可逆，我们分阶段来解
	# 1. 先解决最简单的：双方破防
	# 其实都不用解，易知，对角线为DRAW，下三角为WIN，上三角为LOSE
	v[(1,1)][range(k+1), range(k+1)] = DRAW # 对角线特殊，不求解，由对称性得到；真的求解的话，结果还可能错误
	bobozan(no_defence=(1,1)) # 只求解真·下三角
	# 补全上三角
	for state_a in range(k+1):
		for state_b in range(state_a+1, k+1):
			v[(1,1)][(state_a, state_b)] = WIN + LOSE - v[(1,1)][(state_b, state_a)]
	print_v_and_p(no_defence=(1,1))

	# 2. 再解决次等的：我破防，对方未破防，但是我的气更多（下三角）
	# 上三角部分不用解，易知是LOSE。对方不需要考虑防御，采用跟双方破防一样的战术即可
	v[(1,0)][np.triu_indices_from(v[(0,0)], k=1)] = LOSE
	bobozan(no_defence=(1,0)) # 下三角包括对角线都需要求解
	# TODO：assert 对角线 <= 0.5：气一样多，就算对方不用防御也至少能达到DRAW，而对方可以防御时己方一定落于下风
	# 但是现在解出来不是这样？？？
	# 而且v的结果变数也很大，说明有问题
	print_v_and_p(no_defence=(1,0))
	'''
	从头开始捋一捋

	我没有防
	对方吸 我吸不如低级和高级￼我我C￼
	对方低 我吸不如低级，不如高级￼
	对方高 我吸和不如低级 不如高级￼
	对方防 我吸好于低级好于高级

	综上低级总是小于等于高级或者低于吸 应该被淘汰
	我可以吸或者高级 只有两个选择？
	不对吧 我有防的时候逻辑也是这样 但是￼说不通？？？

	对方没有防
	对方吸 我防御不如吸，不如低级和高级￼￼
	对方低 我吸不如低级，不如防御，不如高级
	对方高 我吸和低级 不如高级和防（高级不一定小于防，因为除非对方可能也把我破防）
	综上吸和低级总是小于等于高级和防御 我应该高级或者防 这个似乎说得通

	我没有防￼
	对方高 我吸和低级 不如高级￼
	对方防 我吸好于低级好于高级
	低级总是小于等于吸，所以我应该吸或者高级？

	这样就简化多了
	    对方 高级 防御
	我
	吸。     输   便宜
	高级     看   可能吃亏

	特别的，对方0的时候高级退化成吸
	不然，对方不能攻击只有防御，所以对方一直防御下去？就不合理了
	是不是方案1也能解出类似的答案？看一看
	'''
	exit() # 先解决完了再说吧

	# 不用解：我未破防，对方破防的情况
	# bobozan(no_defence=(0,1))
	# 直接通过(1,0)对称得到v的结果
	for state_a in range(k+1):
		for state_b in range(k+1):
			v[(0,1)][(state_a, state_b)] = WIN + LOSE - v[(1,0)][(state_b, state_a)]
	# print_v_and_p(no_defence=(0,1))

	# 3. 再解决最复杂的：双方未破防
	v[(0,0)][range(k+1), range(k+1)] = DRAW # 对角线特殊，不求解，由对称性得到；真的求解的话，结果还可能错误
	bobozan(no_defence=(0,0)) 
	# 补全上三角
	for state_a in range(k+1):
		for state_b in range(state_a+1, k+1):
			v[(0,0)][(state_a, state_b)] = WIN + LOSE - v[(0,0)][(state_b, state_a)]
	print_v_and_p(no_defence=(0,0))
	# TODO：求解过程中可能就需要用到上三角？所以需要用到get_v函数？？？

if __name__ == '__main__':
	main()
