import  numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
stage="WAKE"
# file="/home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin/analysis/5-fold-8-layer-sorted/fold-0-batch-size-100/{}_GAN.mat".format(stage)
# truth="/home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin/analysis/5-fold-8-layer-sorted/fold-0-batch-size-100/{}/ground_truth.mat".format(stage)
truth="/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/analysis/5-fold/fold-0/sorted/{}/ground_truth.mat".format(stage)
file="/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/analysis/5-fold/fold-0/sorted/{}/{}.mat".format(stage,stage)
def worst_vs_best(file):
    data=scio.loadmat(file)
    eeg=data["x"]
    mmd=data["mmd"]
    # worst vs best
    plt.subplot(2,1,1)
    plt.plot(eeg[-1])
    plt.subplot(2,1,2)
    plt.plot(eeg[0])
    plt.show()
    # plt.savefig("./{}_worst_vs_best".format(stage))
def truth_vs_fake(truth,fake):
    truth = scio.loadmat(truth)
    fake=scio.loadmat(fake)
    mmd_truth = truth["mmd"][0]
    mmd_fake=fake["mmd"][0]
    plt.hist(mmd_truth,bins=40,label="truth")
    plt.hist(mmd_fake,bins=40,label="fake")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # worst_vs_best(file)
    truth_vs_fake(truth,file)