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