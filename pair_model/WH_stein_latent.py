from matplotlib import pyplot
from numpy import linspace, where, array, heaviside, repeat, power
from scipy.integrate import odeint
from sklearn.metrics import auc
from math import log
import random


# make a nice, big figure
pyplot.figure(figsize=(10,10), dpi=100)

# make a subplot for the susceptible, infected and recovered individuals
# axes_s = pyplot.subplot(311, yscale='log', ylim=(0.1,999999))
axes_s = pyplot.subplot(311, title="Within-host model properties")
axes_s.set_ylabel("Peak Viral Load")

axes_i = pyplot.subplot(312, ylim=(0,10))
axes_i.set_ylabel("Latent period (days)")

axes_r = pyplot.subplot(313, ylim=(0,10))
axes_r.set_ylabel("Infectious time (days)")
axes_r.set_xlabel("Initial Viral Load")

steps_per_day = 30

t = linspace(0, 24, num=24*1000)
v0s = linspace(1,500,5000)

r = 5.5
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

def differential_stein(n_stein, t, r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP):
    dV_dt = r*n_stein[0] - n_stein[0]*(((r*n_stein[0])/kV) + kI*n_stein[1] + kN*n_stein[2] + kP*n_stein[3])
    dI_dt = aI*n_stein[0] + bI*(1-(n_stein[1]/KI))
    dN_dt = aN*n_stein[0]*heaviside(t-tN,1)-dN*n_stein[2]
    dP_dt = aP*n_stein[0]*n_stein[3] + c*n_stein[2]*(t-tP)
    return dV_dt, dI_dt, dN_dt, dP_dt

latents = []
peaks = []
infectious_times = []

for j in v0s:
    solution = odeint(differential_stein, (j, 0, n0, p0), t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]

    peaks.append(max(solution[0]))

    lat = None
    infect = 0

    for step in range(len(solution[0])):
        if solution[0][step] > 20000:
            if not lat:
                lat = step
            infect += 1

    if not lat:
        lat = 0

    latents.append(lat / 1000)
    infectious_times.append(infect / 1000)
        




axes_s.plot(v0s, peaks)
axes_i.plot(v0s, latents)
axes_r.plot(v0s, infectious_times)

pyplot.savefig(fname='WH_stein_latent')
pyplot.show()