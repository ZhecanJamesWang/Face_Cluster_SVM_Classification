import numpy as np


# b = []
# for i in range(5):
# 	a = np.matrix(np.arange(9).reshape((3, 3)))
# 	b.append(a)
# b = np.array(b)
# print b.shape
# print np.mean(b, axis = 0)


# import sys
# sys.path.append('/Users/zhecanwang/liblinear/python')
# from liblinearutil import *
# # Read data in LIBSVM format
# y, x = svm_read_problem('/Users/zhecanwang/liblinear/heart_scale')
# m = train(y[:200], x[:200], '-c 4')
# p_label, p_acc, p_val = predict(y[200:], x[200:], m)



# print np.arange(10)
clusterNum = 10
clusters = list(np.arange(clusterNum))

for cluster in clusters:
	print cluster
