import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
bmap = brewer2mpl.get_map("Set2", "qualitative", 7)
colors = bmap.mpl_colors
import scipy.io as scio
# accs=[0.727,0.747,0.7383,0.7342,0.7514]
# f1s=[0.65,0.672,0.6764,0.6583,0.6758]
# Kappas=[0.5960,0.627,0.6173,0.6058, 0.6302]
# Ws=[0.8091,0.8247,0.8387,0.8268,0.8389]
# N1s=[0.3193,0.318,0.3486,0.3279,0.3485]
# N2s=[0.8156,0.8287,0.8191,0.8152,0.8319]
# N3s=[0.559,0.6637,0.6607,0.614,0.6333]
# Rs=[0.717,0.7252,0.7148,0.7078,0.7276]
# bmap = brewer2mpl.get_map('Paired', 'Qualitative', 5)
# a=[]
# for i in range(5):
#     a.append([accs[i],f1s[i],Kappas[i],Ws[i],N1s[i],N2s[i],N3s[i],Rs[i]])
a=scio.loadmat("./edf-5-fold-table.mat")['a'][::2,:]
a[2]=[0.766,0.689,0.6816,0.8401,0.2711,0.8278,0.7881,0.7185]
labels_a = ['base line', 'repeat', 'GAN(cross dataset)', 'GAN']


labels=['acc','f1 score','kappa','WAKE','N1','N2','N3','REM']
x = np.arange(8)  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x-2*width, a[0], width ,label=labels_a[0])
rects2 = ax.bar(x-width, a[1],width,label=labels_a[1])
rects3 = ax.bar(x, a[2], width,label=labels_a[2])
rects4 = ax.bar(x+width, a[3], width,label=labels_a[3])
# rects5 = ax.bar(x +2*width, a[4], width,label=labels_a[4])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('')
ax.set_title('overall performance ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right', handlelength=1.5, fontsize=16)


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