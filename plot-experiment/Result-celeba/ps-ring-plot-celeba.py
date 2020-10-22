import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

name = ['WAN-FL', 'PS-1', 'Ring-1', 'PS-4', 'Ring-4']
x_name = np.arange(len(name))

## dataset celeba
y1 = [52.13, 23.08, 24.91, 28.91, 21.66]
y2 = [90.03, 90.03, 90.10, 90.04, 90.06]

## dataset femnist
# #y1 = [89.91, 4.08, 4.38, 5.04, 3.85]
# y1 = [10, 4.08, 4.38, 5.04, 3.85]
# y2 = [80.05, 80.06, 80.01, 80.04, 80.17]

fig, ax1 = plt.subplots()

## plt bar
ax1.set_ylabel('Clock Time (hour)', color='tab:red', fontsize=26)
# plt.ylim(0, 10)
ax1.bar(name, y1, edgecolor='red', linewidth=2, color='white')
for x, y in zip(x_name, y1):
	plt.text(x-0.4, y, str(y), fontsize=18)
#ax1.yaxis.set_ticks([1, 4, 64, 128])
#plt.legend(loc="upper right")
plt.xticks(fontsize=17)
plt.yticks(fontsize=10)
# ax1.set_ylim(0,10)

## plt plot
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=26)
ax2.plot(name, y2, marker='^', markersize=20, linestyle='--')
ax2.set_ylim(88,91)
plt.yticks(fontsize=10)

plt.savefig('pdf/eval-time-ps-ring-celeba.pdf', bbox_inches='tight')

plt.show()