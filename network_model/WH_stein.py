from matplotlib import pyplot
from numpy import linspace, where, array, heaviside
from scipy.integrate import odeint
from math import log
import random


# make a nice, big figure
pyplot.figure(figsize=(10,10), dpi=100)

# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(111)
axes_s.set_title("Viral load response to varying initial viral loads")
axes_s.set_ylabel("Viral Load (copies/ml)")
axes_s.set_xlabel("Days")

# axes_i = pyplot.subplot(312)
# axes_i.set_ylabel("P(Transmission)")

# axes_r = pyplot.subplot(313)
# axes_r.set_ylabel("P(Transmission)")
# axes_r.set_xlabel("Steps")

t = linspace(0, 12, num=6000)
# y = linspace(1,10000,2)
y = [1,10,100,1000,10000]

r = 8
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

for j in y:
    solution = odeint(differential_stein, (j, 0, n0, p0), t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]
    
    v_max = 400000
    
    lin_infect = [sim/v_max for sim in solution[0]]
    
    threshold_infect = []
    threshold = 10000
    
    for sim in solution[0]:
        if sim>threshold:
            threshold_infect.append(1)
        else:
            threshold_infect.append(0)

#     print(j,len(where(array(solution[2]) > 100)[0]))
    
    axes_s.plot(t, solution[0], label='V0 = ' + str(j))
#     axes_i.plot(t, solution[2], label='V0 = ' + str(j))
    # axes_i.plot(t, lin_infect, label='V0 = ' + str(j))
    # axes_r.plot(t, threshold_infect, label='V0 = ' + str(j) + ' : Infectious duration = ' + str(sum(threshold_infect)))

axes_s.legend()
# axes_i.legend()
# axes_r.legend()

# solution = odeint(differential_TIV, y0, t, args=(alpha, beta, prod, clear))
# solution = [[row[i] for row in solution] for i in range(3)]

# solution2 = odeint(differential_TIV, y1, t, args=(alpha, beta, prod, clear))
# solution2 = [[row[i] for row in solution2] for i in range(3)]

# solution3 = odeint(differential_TIV, y2, t, args=(alpha, beta, prod, clear))
# solution3 = [[row[i] for row in solution3] for i in range(3)]

# plot numerical solution
# axes_r.plot(t, solution[0], color="black")
# axes_i.plot(t, solution[1], color="black")
# axes_s.plot(t, solution[2], color="black")

# axes_r.plot(t, solution2[0], color="blue")
# axes_i.plot(t, solution2[1], color="blue")
# axes_s.plot(t, solution2[2], color="blue")

# axes_r.plot(t, solution3[0], color="red")
# axes_i.plot(t, solution3[1], color="red")
# axes_s.plot(t, solution3[2], color="red")

# pyplot.savefig(fname='WH_stein_small_v0')
pyplot.show()