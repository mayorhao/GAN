import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
from time import time
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import os
from sklearn import manifold, datasets
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as scio
ROOT_PATH="/home/fanjiahao/matlab-code/feature_extraction/output/SS3"
#load fold one data
def get_data(path):
    file_list = glob.glob(os.path.join(path,"*.mat"))
    dataset = []
    labels = []
    for idx, file in enumerate(file_list):
        data = scio.loadmat(file)
        eeg = data['features_final']
        label = data["y_5"][0]
        # remove N2 stage ,cause it will confusion with others
        # N2_index=np.where(label==2)
        # total_index=np.arange(len(eeg))
        # final_index= np.setdiff1d(total_index,N2_index)
        # eeg=eeg[final_index]

        dataset.extend(eeg)
        labels.extend(label.squeeze().tolist())
        print("%d /%d done"%(idx+1,len(file_list)))
    unique_labels=np.unique(labels)
    dataset = np.vstack(dataset)
    final_data=[]
    final_label=[]
    labels = np.asarray(labels)
    for idx, c in enumerate(unique_labels):
        c_list = np.where(labels == c)[0]
        random_idx = np.random.permutation(c_list)[:1000]
        final_data.extend(dataset[random_idx])
        final_label.extend(labels[random_idx])
    final_data = np.vstack(final_data)
    final_label = np.asarray(final_label)
    return final_data,final_label
#load fold 0 GAN data:
def downSample():
    print('hello')
def get_fake_data():
    print('hello')
# def get_data():
#     data=[]
#     labels=[]
#     raw=scio.loadmat(ROOT_PATH)
#     data=raw["features_final"]
#     labels=raw["y_5"]
#     # for idx,stage in enumerate(["WAKE","N1","N2","N3","REM"]):
#     #     stage_data=data_group[stage]
#     #     label=idx
#     #     for n,epoch in enumerate(stage_data):
#     #         data.append(epoch)
#     #         labels.append(label)
#     return data,labels
# def get_data():
#     digits = datasets.load_digits(n_class=6)
#     data = digits.data
#     label = digits.target
#     n_samples, n_features = data.shape
#     return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    x, y = data[:,0], data[:,1]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    # ax=plt.subplot(111)
    #  将数据点分成三部分画，在颜色上有区分度
    for idx,sample in enumerate(data):
        colors=["r","g","b","y","k"]
        labels=["WAKE","N1","N2","N3","REM"]
        # label = labels[label[idx]]
        if label[idx]!=2:
            ax.scatter(x[idx],y[idx],c=colors[label[idx]])

    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')s
    # ax.set_xlabel('X')
    ax.legend()
    plt.show()
def main():
    data, label = get_data(ROOT_PATH)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    main()