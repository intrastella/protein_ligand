import logging
import pathlib
from datetime import datetime
from pathlib import Path

import sklearn
from keras import backend
from tqdm import tqdm

from numba import cuda
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from models.model_utils import calculate_gradient_penalty


cwd = Path().absolute()
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


cuda_C = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.layers = []

        Hparameters = [(100, 48, 12, 12, 1, 1, 12, 12),
                       (48, 24, 5, 5, 5, 4, 1, 1),
                       (24, 12, 4, 5, 5, 5, 2, 1),
                       (12, 3, 8, 4, 6, 11, 1, 1)]

        def convT_relu(normalize=True):
            for (in_c, out_c, k1, k2, d1, d2, s1, s2) in Hparameters:
                self.layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=(k1, k2), dilation=(d1, d2), stride=(s1, s2)))
                if normalize:
                    self.layers.append(nn.BatchNorm2d(out_c, 0.8))
                self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            return self.layers

        self.convT_blocks = nn.Sequential(
            *convT_relu(normalize=True)
        )

    def forward(self, z):
        feat = z.view(z.shape[0], self.latent_dim, 1, 1)
        feat = self.convT_blocks(feat)
        return feat


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = []

        Hparameters = [(4, 6, 1),
                       (4, 1, 2),
                       (5, 7, 4)]

        def conv_relu():
            for (k, d, s) in Hparameters:
                self.layers.append(
                    nn.Conv2d(3, 3, kernel_size=(1, k), dilation=(1, d), stride=(1, s)))
                self.layers.append(nn.AvgPool2d((4, 4), padding=2, stride=1))
                self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            return self.layers

        self.conv_blocks = nn.Sequential(
            *conv_relu()
        )

        self.linear_block = nn.Sequential(
            nn.Linear(738, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, mat):
        mat = self.conv_blocks(mat)
        mat = mat.view(mat.size(0), np.prod(mat.shape[1:]))
        mat = self.linear_block(mat)
        return mat


class GAN:

    def __init__(self,
                 latent_dim: int = 100,
                 batch_size: int = 60,
                 training_steps: int = 200,
                 n_epoch: int = 5,
                 n_critic: int = 5,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999):

        torch.cuda.empty_cache()
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)

        self.dataloader = None

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_epoch = n_epoch
        self.n_critic = n_critic
        self.latent_dim = latent_dim

        # Models
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

        if cuda_C:
            self.generator.cuda()
            self.discriminator.cuda()

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

    def fit(self, data: DataLoader, exp_dir: Path):
        self.dataloader = data
        self.discriminator.train()
        self.generator.train()

        logger.info('Start training:')

        for epoch in range(self.n_epoch):
            self.train(epoch, exp_dir)

        logger.info('Training finished.')

    def train(self, epoch, exp_dir: Path):
        running_loss = 0.0

        for step, mat in tqdm(enumerate(self.dataloader)):
            total_step = len(self.dataloader) * epoch + step

            folder = f'{exp_dir}/runs'
            p = pathlib.Path(folder)
            p.mkdir(parents=True, exist_ok=True)
            tb = SummaryWriter(folder)

            Tensor = torch.cuda.FloatTensor if cuda_C else torch.FloatTensor

            valid = Variable(Tensor(mat.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.size(0), 1).fill_(0.0), requires_grad=False)
            real_lig = Variable(mat.type(Tensor))
            noise = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], self.latent_dim))))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.discriminator.zero_grad()
            real_output = self.discriminator(real_lig)
            errD_real = torch.mean(real_output)
            D_x = real_output.mean().item()

            fake_lig = self.generator(noise)

            fake_output = self.discriminator(fake_lig.detach())
            errD_fake = torch.mean(fake_output)
            D_G_z1 = fake_output.mean().item()
            gradient_penalty = calculate_gradient_penalty(self.discriminator,
                                                          real_lig.data, fake_lig.data,
                                                          device)

            errD = -errD_real + errD_fake + gradient_penalty * 10
            errD.backward()
            # nn.utils.clip_grad_value_(self.discriminator.parameters(), clip_value=1.0)
            self.discriminator_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------

            # if (step + 1) % self.n_critic == 0:
            if (step + 1) % 1 == 0:

                self.generator.zero_grad()
                fake_lig = self.generator(noise)
                fake_output = self.discriminator(fake_lig)
                errG = -torch.mean(fake_output)
                D_G_z2 = fake_output.mean().item()
                errG.backward()
                # nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=1.0)
                self.generator_optimizer.step()

                logger.info(f"[{epoch + 1}/{epoch}][{step + 1}/{len(self.dataloader)}] "
                                             f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                             f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}"
                                             )

            running_loss += errD.item()
            if step % 1 == 0:
                tb.add_scalar('Discriminator_Loss',
                                  running_loss / 1,
                                  total_step)

            '''if total_step % 10 == 0:
                # true - preds
                generator_score = valid
                discriminator_pred = true_discriminator_out
                gen_ROC = sklearn.metrics.roc_auc_score(discriminator_pred, generator_score)
                dis_ROC = sklearn.metrics.roc_auc_score(true_val, discriminator_score)'''

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            Tensor = torch.cuda.FloatTensor if cuda_C else torch.FloatTensor
            noise = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            generated_data = self.generator(noise)
            return generated_data
