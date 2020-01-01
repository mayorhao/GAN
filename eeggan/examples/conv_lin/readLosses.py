import joblib
import numpy as np
import matplotlib.pyplot as plt
i_epoch, losses_d, losses_g = joblib.load(
    "/eeggan/examples/shallow_conv_lin/models/5-fold-8-layer/k_fold_0/N1/Progressive0_.data")
# print('Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f ' % (loss_d[0], loss_d[1], loss_d[2], loss_g))
for i in range(0,len(losses_g)):
    if (i+1)%100==0:
        wd= -losses_d[i][0] -losses_d[i][1]
        print('epoch: %d Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_D:%.3f Loss_G: %.3f wessteian distance: %.3f' % (i,losses_d[i][0], losses_d[i][1], losses_d[i][2], -wd,losses_g[i],wd))
# "/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/models/k_fold_2/N1/Progressive0_899.data"
start=4499
plt.figure(figsize=(10, 15))
plt.subplot(3, 2, 1)
plt.plot(np.asarray(losses_d)[start:, 0], label='Loss Real')
plt.plot(np.asarray(losses_d)[start:, 1], label='Loss Fake')
plt.title('Losses Discriminator')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(np.asarray(losses_d)[start:, 0] + np.asarray(losses_d)[start:, 1] + np.asarray(losses_d)[start:, 2], label='Loss')
plt.title('Loss Discriminator')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(np.asarray(losses_d)[start:, 2], label='Penalty Loss')
plt.title('Penalty')
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(-np.asarray(losses_d)[start:, 0] - np.asarray(losses_d)[start:, 1], label='Wasserstein Distance')
plt.title('Wasserstein Distance')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(np.asarray(losses_g), label='Loss Generator')
plt.title('Loss Generator')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(modelpath, modelname % jobid + '_losses.png'))
plt.show()