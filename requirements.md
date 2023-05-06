# 需要的库

numpy

scipy：从pip安装可能有问题

## mac

可能报错如下
```
...
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed
```

猜想、当前python版本没有scipy？不应该啊
https://www.jianshu.com/p/2003262d80f0
    
实际解决方案：
https://blog.csdn.net/kl_lk/article/details/123246514

## windows

问题：默认安装的numpy中不包含MKL库，scipy的依赖关系没有实现。
https://www.zhihu.com/question/30188492
