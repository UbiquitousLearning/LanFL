import numpy as np
import matplotlib.pyplot as plt

name = ['WAN-FL', 'PS-1', 'Ring-1', 'PS-4', 'Ring-4']
x_name = np.arange(len(name))

## dataset celeba
# y1 = [52.13, 23.08, 24.90, 28.91, 21.66]
# y2 = [90.03, 90.03, 90.10, 90.04, 90.06]

## dataset femnist
#y1 = [89.91, 4.08, 4.38, 5.04, 3.85]
y1 = [10, 4.08, 4.38, 5.04, 3.85]
y2 = [80.05, 80.06, 80.01, 80.04, 80.17]

fig, ax1 = plt.subplots()

## plt bar
ax1.set_ylabel('Clock Time (hour)', color='tab:red', fontsize=26)
# plt.ylim(0, 10)
ax1.bar(name, y1, edgecolor='red', linewidth=2, color='white')
for x, y in zip(x_name, y1):
	if y == 10:
		plt.text(x-0.4, 10, str(89.91), fontsize=18)
	else:
		plt.text(x-0.3, y, str(y), fontsize=18)

plt.xticks(fontsize=17)
plt.yticks(fontsize=10)
ax1.set_ylim(0,10)

## plt plot
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=26)
ax2.plot(name, y2, marker='^', markersize=20, linestyle='--')
ax2.set_ylim(78,81)
plt.yticks(fontsize=10)

plt.savefig('pdf/eval-time-ps-ring-femnist.pdf', bbox_inches='tight')

plt.show()