import numpy as np

covar = np.loadtxt("covariance_matrix.txt", delimiter="\t")
print(np.linalg.inv(covar))