import logging

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


logger = logging.getLogger(__name__)


cuda_C = True if torch.cuda.is_available() else False


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
        self.latent_dim = latent_dim

        # Models
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

        # loss
        self.loss = nn.BCELoss()
        # kl_loss = nn.KLDivLoss(reduction="batchmean")

        if cuda_C:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss.cuda()

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

    def fit(self, data: DataLoader):
        self.dataloader = data
        self.discriminator.train()
        self.generator.train()

        for epoch in range(self.n_epoch):
            self.train(epoch)

    def train(self, epoch):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

        for step, mat in enumerate(self.dataloader):
            total_step = len(self.dataloader) * epoch + step

            tb = SummaryWriter()

            Tensor = torch.cuda.FloatTensor if cuda_C else torch.FloatTensor

            valid = Variable(Tensor(mat.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.size(0), 1).fill_(0.0), requires_grad=False)
            real_lig = Variable(mat.type(Tensor))
            noise = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], self.latent_dim))))

            # -----------------
            #  Train Generator
            # -----------------
            self.generator_optimizer.zero_grad()

            generated_data = self.generator(noise)
            generator_discriminator_out = self.discriminator(generated_data)

            # w/ KL div
            # log_out = F.log_softmax(generator_discriminator_out, dim=0)
            # gen_loss = kl_loss(log_out, true_labels)

            generator_loss = self.loss(generator_discriminator_out, valid)
            generator_loss.backward()
            self.generator_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.discriminator_optimizer.zero_grad()

            true_discriminator_out = self.discriminator(real_lig)
            true_discriminator_loss = self.loss(true_discriminator_out, valid)

            generator_discriminator_out = self.discriminator(generated_data.detach())
            generator_discriminator_loss = self.loss(generator_discriminator_out, fake)

            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()

            # nn.utils.clip_grad_value_(self.discriminator.parameters(), clip_value=1.0)
            self.discriminator_optimizer.step()

            progress_bar.set_description(f"[{epoch + 1}/{epoch}][{step + 1}/{len(self.dataloader)}] ")

            running_loss += discriminator_loss.item()
            if step % 1000 == 0:
                tb.add_scalar('Discriminator_Loss',
                                  running_loss / 1000,
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
