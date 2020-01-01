import  numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
stage="WAKE"
file="/home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin/analysis/5-fold-8-layer-sorted/fold-0-batch-size-100/{}_GAN.mat".format(stage)
truth_normalize="/home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin/analysis/5-fold-8-layer-sorted/fold-0-batch-size-100/N1/ground_truth_normalized.mat"
truth="/home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin/analysis/5-fold-8-layer-sorted/fold-0-batch-size-100/N1/ground_truth.mat"

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
def truth_vs_fake(truth,truth_normalize,fake):
    truth = scio.loadmat(truth)
    fake=scio.loadmat(fake)
    truth_normalize=scio.loadmat(truth_normalize)
    mmd_truth = truth["mmd"][0]
    mmd_fake=fake["mmd"][0]
    mmd_truth_normalize=truth_normalize["mmd"][0]
    plt.hist(mmd_truth,bins=40,label="truth")
    plt.hist(mmd_fake,bins=40,label="fake")
    plt.hist(mmd_truth_normalize,bins=40,label="truth_normalized")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # worst_vs_best()
    truth_vs_fake(truth,truth_normalize,file)