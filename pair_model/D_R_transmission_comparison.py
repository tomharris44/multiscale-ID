from matplotlib import pyplot
from numpy import linspace, log, power

viral_max = 25000

pyplot.figure(figsize=(15,10), dpi=100)

axes_s = pyplot.subplot(111)
axes_s.set_ylabel("Recipient Initial Viral Load")
axes_s.set_xlabel("Donor Viral Load")

steps_per_day = 30

t = linspace(1, 60001, num=60000)
z = [0.0001,0.001,0.01,0.1]
a = [1,10,100,1000]
b = [1,1.1,1.2,1.5,2,5]
c = [10]#[2,5,10,20]

v_max = 50000
# zeta = 2

axes_s.plot(t, 0.01*t, label='Linear')
# axes_s.plot(t, 0.1*4000*log(t), label='Log')
# axes_s.plot(t,0.1*v_max * power(t,10) / (power(t,10) + power(v_max/2,10)), label='Sigmoid')
for zeta in c:
    axes_s.plot(t,0.01*v_max * power(t,zeta) / (power(t,zeta) + power(v_max/2,zeta)),label='Sigmoid : zeta=' + str(zeta))
# axes_s.plot(t,0.1*v_max * power(t+14000,10) / (power(t+14000,10) + power(v_max/2,10)), label='Sigmoid: Early')
# axes_s.plot(t,0.1*v_max * power(t,10) / (power(t,10) + power(v_max/2,10)), label='Sigmoid: Mid')
# axes_s.plot(t,0.1*v_max * power(t-14000,10) / (power(t-14000,10) + power(v_max/2,10)), label='Sigmoid: Late')
# axes_s.plot(t,0.1*v_max * power(t,10) / (power(t,10) + power(v_max/1.2,10)), label='Sigmoid: Late')
axes_s.legend()

def viral2trans(x):
    return x / (viral_max * steps_per_day)
def trans2viral(x):
    return x * (viral_max * steps_per_day)

secax = axes_s.secondary_xaxis('top', functions=(viral2trans, trans2viral))
secax.set_xlabel('P(Transmission)')

pyplot.savefig(fname='D_R_transmission_comparison')
pyplot.show()