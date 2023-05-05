import numpy as np

k = 4 # 3,4

actions = ['energy', 'attack', 'defence']
p = {}

def actor(state):
    if state in p: return p[state]
    
    if state == (0, 0): p[state] = [1,0,0] # 第一步肯定是攒气
    elif state == (k-1, 0): p[state] = [1,0,0] # 趁此机会攒气到必杀
    # elif state == (0, k-1): p[state] = [1,0,0] # 做什么都是一样的，等死
    else: # 非特殊情况
        if k == 2:
            if state == (1, 1) : p[state] = [1/3, 1/3, 1/3]
        elif k == 3:
            # if state == (0, 1) : p[state] = [0.6538, 0, 0.3462]
            if state == (1, 0) : p[state] = [0.6538, 0.3462, 0]

            elif state == (1, 1) : p[state] = [0.3077, 0.2216, 0.4707]
            
            # elif state == (1, 2) : p[state] = [0.2978, 0.4647, 0.4375]
            elif state == (2, 1) : p[state] = [0.2216, 0.249, 0.5294]
            
            elif state == (2, 2) : p[state] = [0.1906, 0.4047, 0.4047]
        elif k == 4:
            # if state == (0, 1): p[state] = [0.676703, 0, 0.323297]
            if state == (1, 0): p[state] = [0.676703, 0.323297, 0]

            # elif state == (0, 2): p[state] = [0.54356, 0, 0.45644]
            elif state == (2, 0): p[state] = [0.54356, 0.45644, 0]

            elif state == (1, 1): p[state] = [0.289099, 0.167702, 0.543198]
            
            elif state == (2, 1): p[state] = [0.235081, 0.22726, 0.537659]
            elif state == (3, 1): p[state] = [0.172208, 0.226646, 0.601145]
            # elif state == (1, 2): p[state] = []
            # elif state == (1, 3): p[state] = []

            elif state == (2, 2): p[state] = [0.186511, 0.209368, 0.604121]

            elif state == (3, 2): p[state] = [0.120108, 0.226458, 0.653434]
            # elif state == (2, 3): p[state] = []

            elif state == (3, 3): p[state] = [0.147691, 0.426155, 0.426155]

    assert abs(np.sum(p[state]) - 1) < 1e-3
    return p[state]

'''
可以发现，(k-1,k-1)时，p(attack) = p(defence)
k-1, k-1：平
对方energy：
    energy：平：转移到0, 0
    attack：赢
    defence：输
    1/2 = p(energy)*1/2 + p(attack)*1 + p(defence)*0
    1 = p(energy) + p(attack) + p(defence)
    所以p(attack) = p(defence)
对方attack：
    energy：输
    attack：平：转移到k-2，k-2
    defence：v优：转移到k-1，k-2
对方defence
    energy：赢
    attack：v劣：转移到k-2，k-1 = 1-v优
    defence：平：转移到k-1，k-1

1/2
= p(energy)*0 + p(attack)*1/2 + p(attack)*v优
= p(energy)*1 + p(attack)*(1-v优) + p(attack)*1/2
两个方程相加：
1
= p(energy) + p(attack) + p(attack)
也就是概率守恒：p(energy) + p(attack)*2 = 1
所以虽然有三个方程但是不满秩，其实是两个方程
三个未知数：p(energy)，p(attack)，v优

可以表出 v优 即 v<k-1，k-2>
= (1/2 - p(attack)*1/2) / p(attack)
= (1-p(attack)) / (2p(attack)) > 1/2
仅此而已
'''

v = {}

def opposite(val):
    return 1 - val # 也可以改成 -val
WIN = 1
LOSE = opposite(WIN)
DRAW = (WIN+LOSE) / 2 

def critic(state):
    if state[0] < state[1]: # 只计算a >= b的情况
        other_state = (state[1], state[0])
        return opposite(v[other_state]) # a < b 时取对称

    if state in v: return v[state]

    if state[0] == 'LOSE': v[state] = LOSE # 已经被打死
    elif state[0] == 'WIN': v[state] = WIN # 已经打死对方
    elif state[0] == k and state[1] < k: v[state] = WIN # 已经可以必杀对方
    elif state[0] < k and state[1] == k: v[state] = LOSE # 已经可以被必杀

    elif state[0] == state[1]: v[state] = DRAW # 均势

    # 非特殊情况
    # 1、抄到所有p(a>b)的结果。不然的话要解高次方程
    # 2、通过打印方程，把部分可以算出来的v填上，可以得到打破依赖关系的顺序
    # 3、反过来还可以求解p(a<b)的结果，TODO

    # k = 2
    # [[0.5 0.  0. ]
    # [1.  0.5 0. ]
    # [1.  1.  0.5]]

    # 2个V未知数：<2,1>, <1,0>
    elif k == 3 and state == (2,1): v[state] = np.dot([WIN,WIN,DRAW], actor(state))
    # [[0.5        0.17306086 0.         0.        ]
    # [0.82693914 0.5        0.2647     0.        ]
    # [1.         0.7353     0.5        0.        ]
    # [1.         1.         1.         0.5       ]]

    # 5个V未知数
    elif k == 4 and state == (3,2): v[state] = np.dot([WIN,WIN,DRAW], actor(state))
    elif k == 4 and state == (3,1): v[state] = np.dot([WIN,WIN,critic((3,2))], actor(state))
    elif k == 4 and state == (2,1): v[state] = np.dot([critic((3,2)),WIN,DRAW], actor(state))
    elif k == 4 and state == (2,0): v[state] = np.dot([critic((3,1)),WIN,LOSE], actor(state)) # 最后的lose其实没有必要
    # [[0.5        0.23389188 0.10675806 0.         0.        ]
    # [0.76610812 0.5        0.34563446 0.19640529 0.        ]
    # [0.89324194 0.65436554 0.5        0.326717   0.        ]
    # [1.         0.80359471 0.673283   0.5        0.        ]
    # [1.         1.         1.         1.         0.5       ]]

    # 9个V未知数
    # 0.5
    # ?   0.5
    # ?   ?   0.5
    # ?   ?   ?   0.5
    # 1   ?   ?   ?   0.5
    # 1   1   1   1   1   0.5
    # 猜想
    # 0.5
    # 0.6 0.5
    # 0.7 0.6 0.5
    # 0.9 0.7 0.6 0.5
    # 1   0.8 0.7 0.6 0.5
    # 1   1   1   1   1   0.5

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
                print('*', actor(state)[action_a_index], end=' + ')
            print(';')

        # 实际求解
        for action_b in actions: # 假设对方的行动
            if action_b == 'attack' and state[1] == 0: continue # 对方攻击不合法

            # 对自己的不同的行动概率，点乘求和
            v_temp = 0
            for action_a_index, action_a in enumerate(actions):
                if action_a == 'attack' and state[0] == 0:
                    assert actor(state)[action_a_index] == 0 # 己方攻击不合法
                if actor(state)[action_a_index] == 0: # 反正贡献都是0，直接跳过critic部分，不然可能死循环
                    continue
                if state[1] == 0 and actions[action_a_index] == 2: # 特殊情况：对方没有气时，完全不用考虑防御
                    continue

                done, next_state = transfer(state, action_a, action_b)
                v_temp += critic(next_state) * actor(state)[action_a_index]
            
            # print(state, action_b, v_cur_action_b, v_temp)
            assert v_cur_action_b == -1 or abs(v_cur_action_b-v_temp) < 1e-3 # v与action_b无关，殊途同归，从而可以解方程
            if v_cur_action_b == -1: v_cur_action_b = v_temp
        
        v[state] = v_cur_action_b

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
    # 求解所有的v
    v_array = np.zeros((k+1, k+1))
    for state_a in range(k, -1, -1):
        for state_b in range(k, -1, -1):
            state = (state_a, state_b) # list is not hashable
            v_array[state] = critic(state)
    print(v_array)

if __name__ == '__main__':
    main()
