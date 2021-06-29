from matplotlib import pyplot
from numpy import linspace
from scipy.integrate import odeint
from math import log
import random


# make a nice, big figure
pyplot.figure(figsize=(10,10), dpi=100)

initital_conditions = {
    "t": [25000000],
    "i": [0],
    "v": [1],
    "time": [0.0],
}

# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(311)
axes_s.set_ylabel("viral load")

axes_i = pyplot.subplot(312)
axes_i.set_ylabel("infected cells")

axes_r = pyplot.subplot(313)
axes_r.set_ylabel("target cells")
axes_r.set_xlabel("time (arbitrary units)")

t = linspace(0, 14, num=200)
y0 = (25000000, 0, 1)
y1 = (25000000, 0, 1000)
y2 = (25000000, 0, 100000)
alpha = 0.0000022
beta = 1.9
prod = 1.2
clear = 2.78

def differential_TIV(n_TIV, t, alpha, beta, prod, clear):
    dT_dt = -1 * alpha * n_TIV[0] * n_TIV[2]
    dI_dt = alpha * n_TIV[0] * n_TIV[2] - beta * n_TIV[1]
    dV_dt = prod * n_TIV[1] - clear * n_TIV[2]
    return dT_dt, dI_dt, dV_dt

solution = odeint(differential_TIV, y0, t, args=(alpha, beta, prod, clear))
solution = [[row[i] for row in solution] for i in range(3)]

solution2 = odeint(differential_TIV, y1, t, args=(alpha, beta, prod, clear))
solution2 = [[row[i] for row in solution2] for i in range(3)]

solution3 = odeint(differential_TIV, y2, t, args=(alpha, beta, prod, clear))
solution3 = [[row[i] for row in solution3] for i in range(3)]

# plot numerical solution
axes_r.plot(t, solution[0], color="black")
axes_i.plot(t, solution[1], color="black")
axes_s.plot(t, solution[2], color="black")

axes_r.plot(t, solution2[0], color="blue")
axes_i.plot(t, solution2[1], color="blue")
axes_s.plot(t, solution2[2], color="blue")

axes_r.plot(t, solution3[0], color="red")
axes_i.plot(t, solution3[1], color="red")
axes_s.plot(t, solution3[2], color="red")

pyplot.show()
