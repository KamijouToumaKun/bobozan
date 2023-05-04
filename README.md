# Bobozan
solve bobozan game strategy with Nash equilibrium

## 波波攒游戏规则
假设每回合有四类可选的动作：
* 聚气（energy）
* 攻击（attackx，其中x是指需要消耗的气数，气数越高的攻击越厉害）
* 防御（defence）
* 必杀技（防御也防不住。代码文件中，k=x，约定必杀技需要的气数）

## 当只有一个攻击类动作时
solve.py 可以用方程组的方式收敛到唯一的混合策略纳什均衡，速度较快

bobozan_print.py 输出唯一的混合策略纳什均衡所满足的方程组

bobozan_verify.py 给定不同state下的动作概率，代入方程组来验证是否真的收敛到了纳什均衡

## 当有多个攻击类动作时
solve_multiple.py 可以用线性规划的方式随机收敛到某一个混合策略纳什均衡，速度较慢

类似于强化学习minimax-q算法，只是所有state以同样的频率访问到，并且更新

但是我真的尝试强化学习minimax-q时，结果却很不稳定。可能是因为所有state并不是以同样的频率访问到？
