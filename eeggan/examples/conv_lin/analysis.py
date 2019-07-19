import matplotlib
import numpy as np
from torch.autograd import Variable
import torch
import os
from eeggan.examples.conv_lin.model import Generator,Discriminator
model_path='./models/GAN_debug/PAPERFIN4_BhNoMoSc1_FFC4h_WGAN_adaptlambclamp_CONV_LIN_10l_run0/Progressive0.gen'
n_z=200
task_ind=0
rng = np.random.RandomState(task_ind)
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32)
generator = Generator(1,n_z)
generator.load_model(os.path.join(model_path))
z_vars = Variable(torch.from_numpy(1000),volatile=True).cuda()
batch_fake = generator(z_vars)

