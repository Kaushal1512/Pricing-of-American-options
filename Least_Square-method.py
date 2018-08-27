import numpy as np
import math

# Function to generate Paths
def generate_paths(paths, n_paths, p_length):
	for j in range(n_paths):
		paths[j][0] = 1.00

	paths[0][1] = 1.09
	paths[1][1] = 1.16
	paths[2][1] = 1.22
	paths[3][1] = 0.93
	paths[4][1] = 1.11
	paths[5][1] = 0.76
	paths[6][1] = 0.92
	paths[7][1] = 0.88

	paths[0][2] = 1.08
	paths[1][2] = 1.26
	paths[2][2] = 1.07
	paths[3][2] = 0.97
	paths[4][2] = 1.56
	paths[5][2] = 0.77
	paths[6][2] = 0.84
	paths[7][2] = 1.22

	paths[0][3] = 1.34
	paths[1][3] = 1.54
	paths[2][3] = 1.03
	paths[3][3] = 0.92
	paths[4][3] = 1.52
	paths[5][3] = 0.90
	paths[6][3] = 1.01
	paths[7][3] = 1.34

# Function to calculcate A0, A1, A2, ... using Matrix Multiplication
def regression(Y, X, reg_length):
	A = np.zeros((1, reg_length), dtype=float)
	B = np.zeros((reg_length, reg_length), dtype=float)
	C = np.zeros((reg_length, 1), dtype=float)
	l_X = len(X)

	for i in range(reg_length):
		for j in range(reg_length):
			tempSum = 0
			for k in range(l_X):
				tempSum += X[k]**(i + j)
			B[i][j] = tempSum
	
	for j in range(reg_length):
		tempSum = 0
		for k in range(l_X):
			tempSum += Y[k] * (X[k]**j)
		C[j] = tempSum

	A = np.matmul(np.linalg.inv(B), C)
	return A

# Mathematical Function Y = A0 + A1 X + A2 X X
def estimateFunction(x, A):
	tempSum = 0
	for i in range(len(A)):
		tempSum += A[i] * (x ** i)
	return tempSum


# Main Method
if __name__ == "__main__":
	n_paths = 8
	p_length = 4

	S_zero = 1.00
	K = 1.1
	r = 0.06
	t = 1


	paths = np.zeros((n_paths, p_length), dtype=float)

	generate_paths(paths, n_paths, p_length)

	print "========================================================"
	print "Generated Paths:"
	print paths
	print "========================================================"
	print "\n\n"

	payoff = np.zeros((n_paths, p_length), dtype=float)
	cash_flow = np.zeros((n_paths, p_length), dtype=float)
	stopping_rule = np.zeros((n_paths, p_length), dtype=int)

	for j in range(p_length):
		i = p_length - j - 1
		for k in range(n_paths):
			payoff[k][i] = max(0, K - paths[k][i])

	for j in range(n_paths):
		cash_flow[j][p_length - 1] = max(0, K - paths[j][p_length - 1])

	for j in range(p_length - 2):
		i = p_length - 2 - j
		indicies = []
		X = []
		Y = []
		for k in range(n_paths):
			if (payoff[k][i] > 0):
				X.append(paths[k][i])
				Y.append(cash_flow[k][i + 1] * math.exp(- r * t))
				indicies.append(k)

		A = regression(Y, X, 3)

		for k in range(len(indicies)):
			if estimateFunction(paths[indicies[k]][i], A) < payoff[indicies[k]][i]:
				cash_flow[indicies[k]][i] = payoff[indicies[k]][i]
				for y in range(i + 1, p_length):
					cash_flow[indicies[k]][y] = 0
			else:
				cash_flow[indicies[k]][i] = 0

	tempSum = 0
	for j in range(n_paths):
		for i in range(p_length):
			if (cash_flow[j][i] > 0):
				stopping_rule[j][i] = 1
				tempSum += cash_flow[j][i] * math.exp(- r * i)
	
	option_price = tempSum/n_paths

	print "========================================================"
	print "Final Cash Flow:"
	print cash_flow
	print "========================================================"
	print "\n\n"

	print "========================================================"
	print "Stopping Rule:"
	print stopping_rule
	print "========================================================"
	print "\n\n"

	print "Option Price:", option_price