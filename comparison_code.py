import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 3
means_frank = (np.array([-163.8224, -147.9232, -138.568 , -135.244 ])+200)/78
means_guido = (np.array([-148.1548, -144.9672, -128.8952, -129.6744])+200)/78
means_steve = (np.array([-128.6576, -126.6848, -123.6196, -124.6404])+200)/78

means = np.array([means_frank,means_guido,means_steve])
means = means.reshape(-1,3)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

for i in range(4):
    rects1 = plt.bar(index + bar_width*i, tuple(list(means[i])), bar_width,
                     alpha=opacity,
                     label='Iteration ' + str(i))
'''rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Initial')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Bootstrap')

rects3 = plt.bar(index + bar_width, means_steve, bar_width,
alpha=opacity,
color='r',
label='Bootstrap')'''

plt.axhline(y=1.0, color='black', linestyle='-', label = 'Expert')
plt.axhline(y=(200-130)/78, color='black', linestyle='--', label = 'Noisy Expert')

plt.xlabel('Number of Labeled Trajectories')
plt.ylabel('Performance')
plt.title('Mountain Car')
plt.xticks(index + bar_width, ('1', '2', '3'))
plt.legend(loc="upper left", bbox_to_anchor=(1,1))

plt.tight_layout()
plt.show()