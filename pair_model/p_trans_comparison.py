from matplotlib import pyplot
from numpy import linspace, log, power, repeat

viral_max = 50000
steps_per_day = 30

ptrans_max = 50000 / (viral_max*steps_per_day)

pyplot.figure(figsize=(15,10), dpi=100)

axes_s = pyplot.subplot(111, title='P(Transmission) function comparison')
axes_s.set_ylabel("P(Transmission)")
axes_s.set_xlabel("Viral Load")

t = linspace(1, 60001, num=60000)
con = repeat(ptrans_max/2,60000)


axes_s.plot(t, t / (viral_max*steps_per_day), label='Linear')
axes_s.plot(t, con, label='Constant')
axes_s.plot(t, (50000 * power(t,10) / (power(t,10) + power(50000/2,10))) / (viral_max*steps_per_day),label='Sigmoid')
axes_s.legend()

axes_s.axvspan(0, 5000, alpha=0.5, color='red')

pyplot.savefig(fname='p_trans_comparison')
pyplot.show()