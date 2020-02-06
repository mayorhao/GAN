import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
bmap = brewer2mpl.get_map("Set2", "qualitative", 7)
colors = bmap.mpl_colors
import scipy.io as scio

results=[
[0.7005,0.6427,0.598, 0.7655,  0.2793,  0.7773,  0.763  , 0.6286],
[0.7277, 0.6678,0.6384, 0.8137, 0.2847, 0.7819, 0.8079, 0.6508],
[0.7141,0.6574,0.6202, 0.7619,  0.2814,  0.7875,  0.8008 ,0.6552],
[0.7571, 0.6773,0.6682,0.8166 ,0.268 , 0.8234 ,0.7803, 0.6983],
[ 0.7530, 0.6824,0.6671,0.8259 ,0.2736, 0.8181 ,0.8059 ,0.6883],
[0.7456,0.6741,0.6560,0.811 , 0.2885, 0.812,  0.7662 ,0.6927]
]
results=np.array(results)
a=results
labels_a = ['Base Line', 'Reapet', 'MC','SR', 'Transfer','GAN']

labels=['ACC','MF1','Kappa','F1(WAKE)','F1(N1)','F1(N2)','F1(N3)','F1(REM)']
x = np.arange(8)  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x-2*width, a[0], width ,label=labels_a[0])
rects2 = ax.bar(x-width, a[1],width,label=labels_a[1])
rects3 = ax.bar(x, a[2], width,label=labels_a[2])
rects4 = ax.bar(x+width, a[3], width,label=labels_a[3])
rects5 = ax.bar(x+2*width, a[4], width,label=labels_a[4])
rects6 = ax.bar(x+3*width, a[5], width,label=labels_a[5])

# rects5 = ax.bar(x +2*width, a[4], width,label=labels_a[4])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('')
ax.set_title('Comparision of performances in Sleep-EDF ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right', handlelength=1.5, fontsize=12)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
fig.tight_layout()

plt.show()