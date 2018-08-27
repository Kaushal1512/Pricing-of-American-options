# Binomial Tree Implementation of American Option Pricing

import numpy as np

n = 3
stock_price = np.zeros((2**n, ), dtype=float)
u = 2
d = 0.5
r = 0.25

q_u = (1+r - d)/(u - d)
q_d = (u - 1-r)/(u - d)

stock_price[0] = 4
strike_price = 5

intrinsic_payoff = np.zeros((2**n, ), dtype=float)
option_value = np.zeros((2**n, ), dtype=float)

for i in range(1, 2**n):
	if i % 2 == 1:
		stock_price[i] = stock_price[int((i + 1)/2) - 1] * d
	else:
		stock_price[i] = stock_price[int(i/2) - 1] * u
	intrinsic_payoff[i] = max(0, strike_price - stock_price[i])
	if i >= 2**(n - 1) - 1:
		option_value[i] = intrinsic_payoff[i]

def print_Tree(arr, n):
	for i in range(n):
		for j in range(n - i - 1):
			print "\t",
		for j in range(int(2**i) - 1, 2**(i+1) - 1):
			print arr[j], "\t",
		print ""

for i in range(1, 2**(n - 1)):
	j = 2**(n - 1) - i - 1
	expected_value = (q_d * option_value[2*(j + 1) - 1] + q_u * option_value[2*(j + 1)])/(1 + r)
	option_value[j] = max(intrinsic_payoff[j], expected_value)

print "================\t\tStock Price\t\t================"
print_Tree(stock_price, n)
print "========================================================================"
print ""
print ""
print "================\t\tIntrinsic Price\t\t================"
print_Tree(intrinsic_payoff, n)
print "========================================================================"
print ""
print ""
print "================\t\tOption Price\t\t================"
print_Tree(option_value, n)
print "========================================================================"