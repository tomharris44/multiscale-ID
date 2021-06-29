import seaborn as sb
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.metrics import auc
from matplotlib import pyplot

pyplot.figure(figsize=(10,10), dpi=100)

inoc_max = 50000
sig_init = 10
inoc_max_sig = np.power(inoc_max/2,sig_init)

steps_per_day = 100

t = np.linspace(0, 24, num=24 * steps_per_day)

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
    dN_dt = aN*n_stein[0]*np.heaviside(t-tN,1)-dN*n_stein[2]
    dP_dt = aP*n_stein[0]*n_stein[3] + c*n_stein[2]*(t-tP)
    return dV_dt, dI_dt, dN_dt, dP_dt


# span of potential donor initial viral loads
donor_inits = np.linspace(0,5000,num=101)

# span of potential recipient initial viral loads
rec_inits = np.linspace(0,5000,num=101)

# recipient -> donor viral load at transmission
def getVdTrans(rec_init):
    # beta = inoc_max
    # return np.power(rec_init*inoc_max_sig/(beta * (1 - (rec_init/beta))),1/sig_init)
    return rec_init / 0.1

# span of donor viral loads at transmission
donor_trans = [getVdTrans(i) for i in rec_inits]


# donor init and donor trans -> p(trans)
def getPtrans(init,trans,next_trans):
    solution = odeint(differential_stein, (init, 0, n0, p0), t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]

    for i in range(len(solution[0])):
        temp = solution[0][i]
        if temp < trans or temp > next_trans:
            solution[0][i] = 0
    
    return auc(t,solution[0])

# print(getPtrans(400,4000,4100))


# 2d span of p(trans)
ptrans = []
# for i in range(len(donor_inits)):
#     ptrans.append([])

# for each donor viral load at transmission span and span of donor viral loads at transmission -> p(trans) AUC
for i in range(len(donor_inits)):
    print(i)
    for j in range(len(donor_trans)-1):
        ptrans.append((donor_inits[i],rec_inits[j],getPtrans(donor_inits[i],donor_trans[j],donor_trans[j+1])))
        # ptrans[i].append(getPtrans(donor_inits[i],donor_trans[j],donor_trans[j+1]))

# print(ptrans)

res = pd.DataFrame()
res['donor'] = [i[0] for i in ptrans]
res['rec'] = [i[1] for i in ptrans]
res['prob'] = [i[2] for i in ptrans]

print(res)

res = res.pivot('rec','donor','prob')
res = res.iloc[::-1]

print(res)


ax = sb.heatmap(res)

pyplot.show()
