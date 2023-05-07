# Bobozan
solve bobozan game strategy with Nash equilibrium

## 波波攒游戏规则
假设每回合有四类可选的动作：
* 聚气（energy）
* 攻击（attackx，其中x是指需要消耗的气数，气数越高的攻击越厉害）
* 防御（defence）
* 必杀技（防御也防不住。代码文件中，k=x，约定必杀技需要的气数）

## 当只有一个攻击类动作时

### 当该动作耗气为1时，只有唯一的混合策略纳什均衡

证明：https://www.zhihu.com/question/275344377/answer/380364234

### 当该动作耗气大于1时，可能有多个混合策略纳什均衡

solve.py 可以用方程组的方式收敛到混合策略纳什均衡，速度较快

print_equations.py 输出混合策略纳什均衡所满足的方程组

verify.py 给定不同state下的动作概率，代入方程组来验证是否真的收敛到了纳什均衡

k=3和4时的概率值来自：https://www.zhihu.com/question/275344377/answer/579353104 见回答和评论区

## 当有多个攻击类动作时

### 容易想到，存在多个纳什均衡：

例如同时存在attack1和attack2

我们令attack1之外的动作概率都是0，相当于没有其他的动作，此时用solve.py联立方程组能解出一个纳什均衡

我们令attack2之外的动作概率都是0，相当于没有其他的动作，此时用solve.py联立方程组也能解出一个（其实是多个）纳什均衡

我们对各attack动作做混合的假设，也能解出一个纳什均衡

### 线性规划解法

此时不能再用 solve.py 联立方程组，因为得到的矩阵不满秩，对应多个解。而方程组又不方便限定变量的取值范围，给出的解可能很不合理。

solve_multiple.py 可以用线性规划的方式随机收敛到某一个混合策略纳什均衡，速度较慢

类似于强化学习minimax-q算法，只是所有state以同样的频率访问到，并且更新

## 进一步的修改：solve_multiple2.py

### 加入破防类动作

在动作中加入了破防（breakthroughx，其中x是指需要消耗的气数），可以被防御住，但是之后对方不能再使用防御了

加上破防之后，概率部分收敛到的结果更不唯一了

例如必杀k=8，破防耗气为5，此时<6,0>也是必胜的，但获胜途径有多条

* 可以直接攻击破防，如果还没死的话就再攻击一次而胜利

* 也可以吸收变成<7,1>，然后再按照上述方案走：同样是必胜之势，只是对方可能多拖一轮

本文件算出来，吸气和破防的概率一般是五五开的，如：[0.53 0.   0.   0.   0.47 0.  ]

### 简化局势

solve_multiple.py 有时候解出来的概率同是最优的，但是看起来是在“浪”

例如，当对方的气不够最低攻击限度时，己方不应该选择无谓的防御，即p(defence)应该为0，在 solve.py 中就是这么做的

而“浪”的选择中，可能会选择一两次无谓的防御，只是最后仍能获得胜利

例如 k=8, attack仅有5的情况，当双方的气不到5时，双方都应该努力攒气才对

* 而这次跑出来的结果，收敛得到的 p[2,0] 不是 [1,0,0] 

```
v:

[[0.5        0.14889064 0.66809442 0.71629508 0.99608092 0.30260185
  0.60176555 0.67553492 0.74460753]

[0.99601745 0.5        0.27491328 0.29201751 0.60540749 0.61425782
  0.83858238 0.04199693 0.45595999]

[0.99601746 0.99601745 0.5        0.2293886  0.92328923 0.97013374
  0.86650954 0.60234831 0.10405566]

[0.99999999 0.99601746 0.99601746 0.5        0.88580561 0.81109345
  0.90357999 0.02553662 0.67033609]

[0.99999999 0.99999999 0.99601746 0.99601746 0.5        0.79749676
  0.27923375 0.37053706 0.71568522]

[0.99999999 1.         1.         0.99601746 0.99608092 0.5
  0.22120403 0.53796232 0.13496775]

[0.99999999 1.         1.         1.         0.97311846 0.76120663
  0.5        0.01943865 0.66837701]

[1.         1.         1.         1.         1.         0.97116366
  0.88579747 0.5        0.29883698]

[1.         1.         1.         1.         1.         1.
  1.         1.         0.5       ]]

p:

-1,-1,-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,-1,

[0.55 0.   0.45],[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],[0.55 0.   0.45],[1. 0. 0.],-1,-1,-1,-1,-1,-1,

[0.5 0.  0.5],[1. 0. 0.],[0.55 0.   0.45],[1. 0. 0.],-1,-1,-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0. 0. 1.],[0.   0.99 0.01],-1,-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.93 0.07 0.  ],[0.24 0.34 0.42],-1,-1,-1,

-1,[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.03 0.72 0.25],[0.11 0.66 0.23],-1,-1,

-1,-1,-1,-1,-1,-1,-1,-1,-1,
```

* 又如这次跑出来的结果，收敛到不同的纳什均衡，且得到的 p[4,0] 不是 [1,0,0] 

```
[[0.5        0.99689932 0.10369555 0.81159096 0.19745167 0.92014893
  0.28813071 0.95220512 0.05310128]

[0.73249175 0.5        0.4964996  0.19587621 0.48185376 0.2981694
  0.61792651 0.95516981 0.58716766]

[0.92857766 0.73249175 0.5        0.33306888 0.4917762  0.78668369
  0.62257974 0.07920253 0.11467094]

[0.99999999 0.92857766 0.73249175 0.5        0.91422945 0.62882522
  0.8047908  0.27472521 0.77127297]

[1.         0.99999999 0.92857766 0.73249175 0.5        0.66250753
  0.49652694 0.25055156 0.73918025]

[1.         1.         0.99999999 0.92857766 0.73249175 0.5
  0.88618599 0.60260141 0.22623466]

[1.         1.         1.         1.         0.8849877  0.65624713
  0.5        0.58819187 0.69899801]

[1.         1.         1.         1.         1.         0.85217521
  0.74751596 0.5        0.52028988]

[1.         1.         1.         1.         1.         1.
  1.         1.         0.5       ]]

-1,-1,-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,-1,

[0.5 0.  0.5],[1. 0. 0.],[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,

[0.5 0.  0.5],[0.63 0.   0.37],[1. 0. 0.],[0.62 0.38 0.  ],[0.78 0.22 0.  ],-1,-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.78 0.22 0.  ],[0.3  0.16 0.54],-1,-1,-1,

-1,[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.13 0.29 0.59],[0.16 0.33 0.5 ],-1,-1,

-1,-1,-1,-1,-1,-1,-1,-1,-1,
```

为了简化局势，solve_multiple2.py 提供了 flag_min_action_attack_energy_cost 这一选项，默认是关闭的

如果打开，则求解时加入人为限制，不考虑defence这一选项，p中直接少了这一维度

### 修改BUG

solve_multiple.py 中的BUG：当动作中有耗气>1的时候，相比起原来 solve.py 求解的states，需要求解更多state

所以 solve_multiple.py 求解出来的v有很大随机性，这不代表收敛到不同的纳什均衡，而是其答案是错误的！

solve_multiple2.py 中修改了这一BUG，同时尝试了整个v矩阵初始化和求解，仅对下三角v初始化和求解，仅对原始的 solve.py 中那些states求解等三种方案

其中，方案1应该是最正确的

方案2中，因为此时状态转移不保证始终有state[0]>=state[1]，但是我们只对下三角的v进行初始化和求解，这样一来，当状态转移到上三角时，取到的v值仍是错的。

奇怪的是，虽然取到的v值仍是错的，但也能给出比较稳定的结果，而且三种方案的结果一致，所以应该是对的（虽然仍能保证）

* 上述 k=8, attack仅有5的情况

```
v=
[[ 0.5        -1.         -1.         -1.         -1.         -1.
  -1.         -1.         -1.        ]
 [ 0.6279577   0.5        -1.         -1.         -1.         -1.
  -1.         -1.         -1.        ]
 [ 0.80835305  0.6279577   0.5        -1.         -1.         -1.
  -1.         -1.         -1.        ]
 [ 1.          0.80835305  0.6279577   0.5        -1.         -1.
  -1.         -1.         -1.        ]
 [ 1.          1.          0.80835305  0.6279577   0.5        -1.
  -1.         -1.         -1.        ]
 [ 1.          1.          1.          0.80835305  0.6279577   0.5
  -1.         -1.         -1.        ]
 [ 1.          1.          1.          1.          0.78804256  0.59137301
   0.5        -1.         -1.        ]
 [ 1.          1.          1.          1.          1.          0.76292818
   0.67868453  0.5        -1.        ]
 [ 1.          1.          1.          1.          1.          1.
   1.          1.          0.5       ]]

p=
-1,-1,-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,-1,-1,

[1. 0. 0.],[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,-1,

[0.5 0.  0.5],[1. 0. 0.],[1. 0. 0.],[1. 0. 0.],-1,-1,-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.9 0.1 0. ],[0.91 0.09 0.  ],-1,-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.89 0.11 0.  ],[0.39 0.04 0.56],-1,-1,-1,

[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[0.5 0.  0.5],[1. 0. 0.],[0.23 0.03 0.74],[0.3  0.06 0.64],-1,-1,

-1,-1,-1,-1,-1,-1,-1,-1,-1,
```

## 强化学习解法

我真的尝试写了强化学习minimax-q，但结果却很不稳定。可能是因为所有state并不是以同样的频率访问到？
