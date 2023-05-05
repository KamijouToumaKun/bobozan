import numpy as np

k = 5

actions = ['energy', 'attack', 'defence']
p = {}

def actor(state):
    if state in p: return p[state]
    
    if state == (0, 0): p[state] = [1,0,0] # 第一步肯定是攒气
    elif state == (k-1, 0): p[state] = [1,0,0] # 趁此机会攒气到必杀
    # elif state == (0, k-1): p[state] = [1,0,0] # 做什么都是一样的，等死
    else:
        p[state] = 'p' + str(state)

    return p[state]

v = {}

def opposite(val_or_str):
    if isinstance(val_or_str, int): return 1 - val_or_str # 也可以改成 -val_or_str
    else: return '(1 - ' + val_or_str + ')'
WIN = 1
LOSE = opposite(WIN)
DRAW = (WIN+LOSE) / 2 

def critic(state):
    if state[0] < state[1]: # 只计算a >= b的情况
        other_state = (state[1], state[0])
        if other_state in v: return opposite(v[other_state])
        else: return opposite('v' + str(other_state)) # a < b 时取对称

    if state in v: return v[state]

    if state[0] == 'LOSE': v[state] = LOSE # 已经被打死
    elif state[0] == 'WIN': v[state] = WIN # 已经打死对方
    elif state[0] == k and state[1] < k: v[state] = WIN # 已经可以必杀对方
    elif state[0] < k and state[1] == k: v[state] = LOSE # 已经可以被必杀

    elif state[0] == state[1]: v[state] = DRAW # 均势

    else:
        v_cur_action_b = -1
        
        print('v' + str(state))
        # 打印方程
        for action_b in actions: # 假设对方的行动
            if action_b == 'attack' and state[1] == 0: continue # 对方攻击不合法
            print('=', end=' ')
            for action_a_index, action_a in enumerate(actions):
                if action_a == 'attack' and state[0] == 0:
                    assert actor(state)[action_a_index] == 0 # 己方攻击不合法
                if actor(state)[action_a_index] == 0: # 反正贡献都是0，直接跳过critic部分，不然可能死循环
                    continue
                if state[1] == 0 and action_a == 'defence': # 特殊情况：对方没有气时，完全不用考虑防御
                    continue

                done, next_state = transfer(state, action_a, action_b)
                # next_state[0] == next_state[1] 这一条此时往往没有record下来
                if done or next_state in v or next_state[0] == next_state[1]: print(critic(next_state), end=' ')
                else: print('v' + str(next_state), end=' ')

                pi = actor(state)
                if isinstance(pi, list):
                    print('*', actor(state)[action_a_index], end=' + ')
                else:
                    print('*', '%s[%d]' % (actor(state), action_a_index), end=' + ')
            print(';')

        v[state] = 'v' + str(state)

    return v[state]

def transfer(state, action_a, action_b):
    if action_a == 'energy':
        if action_b == 'energy': return False, (state[0]+1, state[1]+1)
        elif action_b == 'attack': return True, ('LOSE', 'WIN')
        else: return False, (state[0]+1, state[1])
    elif action_a == 'attack':
        if action_b == 'energy': return True, ('WIN', 'LOSE')
        elif action_b == 'attack': return False, (state[0]-1, state[1]-1)
        else: return False, (state[0]-1, state[1])
    else: # defence
        if action_b == 'energy': return False, (state[0], state[1]+1)
        elif action_b == 'attack': return False, (state[0], state[1]-1)
        else: return False, (state[0], state[1])

def main():
    # 打印所有的v方程
    for state_a in range(k, -1, -1):
        for state_b in range(k, -1, -1):
            state = (state_a, state_b) # list is not hashable
            critic(state)

if __name__ == '__main__':
    main()

'''
k = 3
v(2, 1)
= 1 * p(2, 1)[0] + 1 * p(2, 1)[1] + 0.5 * p(2, 1)[2] + ;
= 0 * p(2, 1)[0] + v(1, 0) * p(2, 1)[1] + 1 * p(2, 1)[2] + ;
= 1 * p(2, 1)[0] + 0.5 * p(2, 1)[1] + v(2, 1) * p(2, 1)[2] + ;

v(1, 0)
= v(2, 1) * p(1, 0)[0] + 1 * p(1, 0)[1] + ;
= 1 * p(1, 0)[0] + 0.5 * p(1, 0)[1] + ;

未知数：5+2 = 7个
p(2, 1)[0,1,2], v(2, 1)
p(1, 0)[0,1], v(1, 0)

方程：5+2 = 7个

矩阵形式：

1 1  .5 -1  0 0 0       p21_0       0
0 v10 1 -1  0 0 0       p21_1       0
1 .5 v21 -1 0 0 0       p21_2       0
1 1  1  0   0 0 0       v21     =   1
0 0  0  0   v21 1 -1    p10_0       0
0 0  0  0   1 0.5 -1    p10_1       0
0 0  0  0   1 1 0       v10         1

这不是一次方程组，而是二次的
也不是严格分块，块与块之间有联系
在消去未知数时还会叠加到更高次，听说k=4会叠加到24次
'''

'''
k = 4
v(3, 2)
= 1 * p(3, 2)[0] + 1 * p(3, 2)[1] + 0.5 * p(3, 2)[2] + ;
= 0 * p(3, 2)[0] + v(2, 1) * p(3, 2)[1] + v(3, 1) * p(3, 2)[2] + ;
= 1 * p(3, 2)[0] + 0.5 * p(3, 2)[1] + v(3, 2) * p(3, 2)[2] + ;

v(3, 1)
= 1 * p(3, 1)[0] + 1 * p(3, 1)[1] + v(3, 2) * p(3, 1)[2] + ;
= 0 * p(3, 1)[0] + v(2, 0) * p(3, 1)[1] + 1 * p(3, 1)[2] + ;
= 1 * p(3, 1)[0] + v(2, 1) * p(3, 1)[1] + v(3, 1) * p(3, 1)[2] + ;

v(2, 1)
= v(3, 2) * p(2, 1)[0] + 1 * p(2, 1)[1] + 0.5 * p(2, 1)[2] + ;
= 0 * p(2, 1)[0] + v(1, 0) * p(2, 1)[1] + v(2, 0) * p(2, 1)[2] + ;
= v(3, 1) * p(2, 1)[0] + 0.5 * p(2, 1)[1] + v(2, 1) * p(2, 1)[2] + ;

v(2, 0)
= v(3, 1) * p(2, 0)[0] + 1 * p(2, 0)[1] + ;
= 1 * p(2, 0)[0] + v(1, 0) * p(2, 0)[1] + ;

v(1, 0)
= v(2, 1) * p(1, 0)[0] + 1 * p(1, 0)[1] + ;
= v(2, 0) * p(1, 0)[0] + 0.5 * p(1, 0)[1] + ;

未知数：13+5 = 18个
p(3, 2)[0,1,2], v(3, 2)
p(3, 1)[0,1,2], v(3, 1)
p(2, 1)[0,1,2], v(2, 1)
p(2, 0)[0,1], v(2, 0)
p(1, 0)[0,1], v(1, 0)

方程：13+5 = 18个
'''

'''
k = 5
v(4, 3)
= 1 * p(4, 3)[0] + 1 * p(4, 3)[1] + 0.5 * p(4, 3)[2] + ;
= 0 * p(4, 3)[0] + v(3, 2) * p(4, 3)[1] + v(4, 2) * p(4, 3)[2] + ;
= 1 * p(4, 3)[0] + 0.5 * p(4, 3)[1] + v(4, 3) * p(4, 3)[2] + ;

v(4, 2)
= 1 * p(4, 2)[0] + 1 * p(4, 2)[1] + v(4, 3) * p(4, 2)[2] + ;
= 0 * p(4, 2)[0] + v(3, 1) * p(4, 2)[1] + v(4, 1) * p(4, 2)[2] + ;
= 1 * p(4, 2)[0] + v(3, 2) * p(4, 2)[1] + v(4, 2) * p(4, 2)[2] + ;

v(4, 1)
= 1 * p(4, 1)[0] + 1 * p(4, 1)[1] + v(4, 2) * p(4, 1)[2] + ;
= 0 * p(4, 1)[0] + v(3, 0) * p(4, 1)[1] + 1 * p(4, 1)[2] + ;
= 1 * p(4, 1)[0] + v(3, 1) * p(4, 1)[1] + v(4, 1) * p(4, 1)[2] + ;

v(3, 2)
= v(4, 3) * p(3, 2)[0] + 1 * p(3, 2)[1] + 0.5 * p(3, 2)[2] + ;
= 0 * p(3, 2)[0] + v(2, 1) * p(3, 2)[1] + v(3, 1) * p(3, 2)[2] + ;
= v(4, 2) * p(3, 2)[0] + 0.5 * p(3, 2)[1] + v(3, 2) * p(3, 2)[2] + ;

v(3, 1)
= v(4, 2) * p(3, 1)[0] + 1 * p(3, 1)[1] + v(3, 2) * p(3, 1)[2] + ;
= 0 * p(3, 1)[0] + v(2, 0) * p(3, 1)[1] + v(3, 0) * p(3, 1)[2] + ;
= v(4, 1) * p(3, 1)[0] + v(2, 1) * p(3, 1)[1] + v(3, 1) * p(3, 1)[2] + ;

v(3, 0)
= v(4, 1) * p(3, 0)[0] + 1 * p(3, 0)[1] + ;
= 1 * p(3, 0)[0] + v(2, 0) * p(3, 0)[1] + ;

v(2, 1)
= v(3, 2) * p(2, 1)[0] + 1 * p(2, 1)[1] + 0.5 * p(2, 1)[2] + ;
= 0 * p(2, 1)[0] + v(1, 0) * p(2, 1)[1] + v(2, 0) * p(2, 1)[2] + ;
= v(3, 1) * p(2, 1)[0] + 0.5 * p(2, 1)[1] + v(2, 1) * p(2, 1)[2] + ;

v(2, 0)
= v(3, 1) * p(2, 0)[0] + 1 * p(2, 0)[1] + ;
= v(3, 0) * p(2, 0)[0] + v(1, 0) * p(2, 0)[1] + ;

v(1, 0)
= v(2, 1) * p(1, 0)[0] + 1 * p(1, 0)[1] + ;
= v(2, 0) * p(1, 0)[0] + 0.5 * p(1, 0)[1] + ;

未知数：24+9 = 33个
p(4, 3)[0,1,2], v(4, 3)
p(4, 2)[0,1,2], v(4, 2)
p(4, 1)[0,1,2], v(4, 1)
p(3, 2)[0,1,2], v(3, 2)
p(3, 1)[0,1,2], v(3, 1)
p(3, 0)[0,1], v(3, 0)
p(2, 1)[0,1,2], v(2, 1)
p(2, 0)[0,1], v(2, 0)
p(1, 0)[0,1], v(1, 0)

方程：24+9 = 33个
'''
