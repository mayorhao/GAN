import joblib
import numpy as np
import matplotlib.pyplot as plt
def readSingleLoss():
    i_epoch, losses_d, losses_g = joblib.load(
        "/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/k_fold_2/REM/Progressive0_.data")
    # print('Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f ' % (loss_d[0], loss_d[1], loss_d[2], loss_g))
    for i in range(0,len(losses_g)):
        if (i+1)%100==0:
            wd= -losses_d[i][0] -losses_d[i][1]
            print('epoch: %d Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_D:%.3f Loss_G: %.3f wessteian distance: %.3f' % (i,losses_d[i][0], losses_d[i][1], losses_d[i][2], -wd,losses_g[i],wd))
    # "/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/k_fold_2/N1/Progressive0_899.data"
    start=4499
    losses_r=np.asarray(losses_d)[::1, 0]
    losses_f=np.asarray(losses_d)[::1,1]
    losses_p=np.asarray(losses_d)[::1,2]
    losses_g=np.asarray(losses_g)[::1]
    # plt.figure(figsize=(10, 15))
    x=np.linspace(0,5500,550)
    plt.plot(x,losses_r, label='Loss Real')
    plt.plot(x,losses_f, label='Loss Fake')
    plt.plot(x,-losses_r-losses_f,label='wd')
    plt.title('Losses Discriminator')
    plt.legend()
    # plt.subplot(3, 2, 2)
    # plt.plot(losses_r + losses_f + losses_p, label='Loss')
    # plt.title('Loss Discriminator')
    # plt.legend()
    # # plt.subplot(3, 2, 3)
    # # plt.plot(np.asarray(losses_d)[start:, 2], label='Penalty Loss')
    # # plt.title('Penalty')
    # plt.legend()
    # plt.subplot(3, 2, 4)
    # plt.plot(-losses_r -losses_f, label='Wasserstein Distance')
    # plt.title('Wasserstein Distance')
    # plt.legend()
    # plt.subplot(3, 2, 5)
    # plt.plot(losses_g, label='Loss Generator')
    # plt.title('Loss Generator')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(modelpath, modelname % jobid + '_losses.png'))
    plt.show()
def readMultiLoss():
    # path="/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/edf-5_fold/k_fold_4/{}/Progressive0_.data"
    paths=["/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/edf-5_fold/k_fold_4/N1/Progressive0_.data","/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/GAN_debug/conv_linear_0_N1/Progressive0_.data",]
    stages=["WAKE","N1","REM","N3"]
    color=["g","y","r","b"]
    paths_label=["simple","complete","8-layer"]
    i_epoches=[]
    losses_d_total=[]
    losses_g_total=[]
    for idx,stage in enumerate(paths):
        file_path=stage
        i_epoch, losses_d, losses_g = joblib.load(file_path)
        losses_r = np.asarray(losses_d)[::10, 0]
        losses_f = np.asarray(losses_d)[::10, 1]
        losses_p=np.asarray(losses_d)[::10, 2]
        x = np.linspace(0, 5500, 550)
        plt.plot(x,losses_r + losses_f+losses_p, label=paths_label[idx],color=color[idx])
    #     i_epoches.append(i_epoch)
    #     losses_d_total.append(losses_d)
    #     losses_g_total.append(losses_g)
    # index=np.arange(len(losses_d_total[0]))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # readSingleLoss()
    readMultiLoss()

