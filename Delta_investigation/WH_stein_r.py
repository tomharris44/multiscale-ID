from matplotlib import pyplot
from numpy import linspace, where, array, heaviside, repeat, power
from scipy.integrate import odeint
from sklearn.metrics import auc
from math import log, ceil
import random


# make a nice, big figure
pyplot.figure(figsize=(10,10), dpi=100)

# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(211, title='Effect of growth parameter r on viral load curve', yscale='log', ylim=(0.1,999999))
# axes_s = pyplot.subplot(211)
axes_s.set_ylabel("Viral Load")

axes_i = pyplot.subplot(212)
axes_i.set_ylabel("P(Transmission)")

# axes_r = pyplot.subplot(313)
# axes_r.set_ylabel("P(Transmission)")
# axes_r.set_xlabel("Days")

steps_per_day = 30

zeta = 10

init_v = 100

detect_threshold = 10000

t = linspace(0, 24, num=24 * steps_per_day)
# y = linspace(1,10000,2)
# y = [1,10,100,1000,10000]
y = [4.5,14.2,45,142.2,450]

r = 5.7
r_delta = 5.5
kI = 0.05
kN = 0.5
kP = 2
aI = 0.000000001
aN = 0.00000001
aP = 0.000005
bI = 2
dN = 0.05
c = 0.01
kV = 100000000000
KI = 100
tN = 2.5
tP = 3
n0 = 0
p0 = 2

rs = [5.5,6,6.5,7]
# init_vs = [1,10,100,1000,10000,100000]

def differential_stein(n_stein, t, r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP):
    dV_dt = r*n_stein[0] - n_stein[0]*(((r*n_stein[0])/kV) + kI*n_stein[1] + kN*n_stein[2] + kP*n_stein[3])
    dI_dt = aI*n_stein[0] + bI*(1-(n_stein[1]/KI))
    dN_dt = aN*n_stein[0]*heaviside(t-tN,1)-dN*n_stein[2]
    dP_dt = aP*n_stein[0]*n_stein[3] + c*n_stein[2]*(t-tP)
    return dV_dt, dI_dt, dN_dt, dP_dt

for j in rs:
    solution = odeint(differential_stein, (init_v, 0, n0, p0), t, args=(j,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]

    area = auc(t,solution[0])

    for i in range(len(solution[0])):
        if solution[0][i] > detect_threshold and solution[0][i-1] <= detect_threshold:
            detected = ceil(i / steps_per_day)
            init_measure = solution[0][detected * steps_per_day]

    print(detected, init_measure)
    print(max(solution[0]))

    v_max = 25000

    lin_infect = [sim/(v_max * steps_per_day) for sim in solution[0]]

    threshold_infect = []
    threshold = 10000
    ptrans = 1.0 / steps_per_day

    for sim in solution[0]:
        if sim>threshold:
            threshold_infect.append(ptrans)
        else:
            threshold_infect.append(0)

    axes_s.plot(t, solution[0], label='r = ' + str(j) + ', AUC = ' + str(round(area,2)))
    axes_i.plot(t, lin_infect, label='r = ' + str(j))
    # axes_r.plot(t, threshold_infect, label='r = ' + str(j) + ' : Infectious duration = ' + str(round(sum(threshold_infect),2)) + ' days')


axes_s.legend()
axes_i.legend()
# axes_r.legend()

pyplot.savefig(fname='WH_stein')
pyplot.show()