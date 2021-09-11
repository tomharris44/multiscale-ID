import numpy as np
import pylab as plt
import matplotlib.pyplot as ptr

x = [2,5,6,6,6,3,7,3,3,5,2,2,8,8,8,6,6,6,6,6,6,6,6,7,7,6,6,6,6,6,6]
y = [4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5]

print(np.mean(x))
print(np.median(x))

ptr.figure(figsize=(10,10), dpi=100)

ax = plt.subplot(111)
ax.boxplot(y, bootstrap=1000, notch=True)

ptr.show()

