# 抽样与蒙地卡罗(一)：随机数生成

## 前言

一般情况下的机器学习训练，我们透过输入的资料来训练模型、验证模型绩效等；然而，大部分实务情况是缺乏资料，或是资料难以生成的，因此近期也不断出现各种改进方法，比方基于小样本的演算法，在近年的ICML (International Conference Machine Learning) 中提出了许多出色的论文，最常见的比如Finetune 在深度学习中的应用等。

另外一种方法是，我们可以基于某种已经确定的方式来生成假的样本，生成的方式实际上是统计学的基础，也就是从一个有限的母体中去产生随机观测。

注意！如果有些人对深度学习的发展熟悉，GAN 等技术与今天要介绍的技术是有点出入的，今天要介绍的随机数产生器着重的目的在于，在我们已经知道目标样本的分布情况下，我们实际只拿到很少部分的资料时，可以透过这种方式来生成随机观测且符合该分布的样本，用以训练模型或是做为预处理使用，亦是许多强化学习模型的基础，是非常好用的工具。

什么叫做随机生成一组样本呢? 举例来说，我们透过python 的numpy random 库来生成几个比较常见的样本看看：

![](images/091a31b5edcca9ef892c84bc2f16e05.png)

```
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

  

class RandomNumberGenerator:

    def __init__(self, num):

        self.num = num

  

    def normal_sample(self, mu, sigma):

        return np.random.normal(loc=mu, scale=sigma, size=self.num)

    def uniform_sample(self, lb, ub):

        return np.random.uniform(low=lb, high=ub, size=self.num)

    def exponential_sample(self, ld):

        return np.random.exponential(scale=ld, size=self.num)

if __name__ == '__main__':

    rng = RandomNumberGenerator(num=1000000)

    nm = rng.normal_sample(mu=0, sigma=1)

    uni = rng.uniform_sample(lb=-1, ub=1)

    ep = rng.exponential_sample(ld=1)

  

    sns.distplot(nm)

    sns.distplot(uni)

    sns.distplot(ep)

  

    plt.title('Random Number Generator')

    plt.legend(['Normal Distribution', 'Uniform Distribution', 'Exponential Distribution'])

    plt.show()
```


简单的分布是好生成的，但许多时候模型所需的样本并不是简单分布，比方说我们想要生成一个基于特殊分布的样本：

![](c1be01f7a02dd56ff430f727171e91f.png)

诸如此类的奇怪分布，有时候并不是这么好直接去取样，因此我们就需要一点方法来调整抽样方式。