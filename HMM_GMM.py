import numpy as np
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture

# 生成一些模拟的连续观测序列数据
np.random.seed(42)
obs_seq = np.random.randn(100, 2)

# 使用GMM拟合连续观测序列数据
n_components = 3
gmm = GaussianMixture(n_components=n_components)
gmm.fit(obs_seq)

# 获取GMM的均值和协方差矩阵
means = gmm.means_
covariances = gmm.covariances_

# 使用HMM拟合连续观测序列数据
n_hidden_states = 2
hmm_model = hmm.GaussianHMM(n_components=n_hidden_states, covariance_type='full')
hmm_model.fit(obs_seq)

# 打印HMM的转移概率矩阵
print("HMM Transition Matrix:")
print(hmm_model.transmat_)

# 打印HMM的隐藏状态分布的均值和协方差矩阵
print("\nHMM Means:")
print(hmm_model.means_)
print("\nHMM Covariances:")
print(hmm_model.covars_)
