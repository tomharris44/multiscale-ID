from matplotlib import pyplot
from numpy import linspace, log, power, repeat

v0s_sig = [25.305359633565697, 44.99999999999998, 80.0225734517515, 142.302494707577, 253.05359633565698, 449.99999999999983] 
v0s_lin = [8.002257345175153, 14.230249470757709, 25.305359633565697, 44.99999999999998, 80.0225734517515, 142.302494707577, 253.05359633565698, 449.99999999999983] 
v0s_rand = [4.500000000000001, 8.002257345175153, 14.230249470757709, 25.305359633565697, 44.99999999999998, 80.0225734517515, 142.302494707577, 253.05359633565698, 449.99999999999983]


pyplot.figure(figsize=(15,10), dpi=100)

axes_s = pyplot.subplot(111, title='Effect of index case initial viral load on decoherence time')
axes_s.set_ylabel("Decoherence time (generations)")
axes_s.set_xlabel("Index case initial viral Load")

sig_results =  [4.72, 4.06, 3.7, 3.02, 2.88, 2.1]
rand_results = [6.4, 4.28, 3.04, 3.72, 3.82, 4.9, 5.38, 6.64, 4.72]
lin_results = [10.46, 6.8, 4.873469387755102, 5.58, 4.34, 5.32, 5.02, 5.58]

axes_s.plot(v0s_sig, sig_results, label='Sigmoid - Sigmoid')
axes_s.plot(v0s_rand, rand_results, label='Sigmoid - Random')
axes_s.plot(v0s_lin, lin_results, label='Sigmoid - Linear')

axes_s.legend()

pyplot.savefig(fname='deco_times')
pyplot.show()