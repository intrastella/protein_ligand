import logging
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

        def linear_relu(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def conv_relu(out_feat, normalize=True):
            layers = [nn.Conv2d(24, 12, (3, 3), stride=(2, 2)), nn.Conv2d(12, 3, (5, 5), dilation=(5, 5))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_block = nn.Sequential(
            *linear_relu(latent_dim, 128, normalize=False),
            *linear_relu(128, 256),
            *linear_relu(256, 512),
        )
        # 274
        self.conv_block = nn.Sequential(
            *conv_relu(81)
        )

    def forward(self, z):
        img = self.linear_block(z)
        img = img.view(z.shape[0], 24, 249, 171)
        img = self.conv_block(img)
        return img


class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN:

    def __init__(self,
                 channels: int = 3,
                 latent_dim: int = 100,
                 h_size: int = 274,
                 w_size: int = 81,
                 batch_size: int = 60,
                 training_steps: int = 200,
                 n_epoch: int = 5,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999):

        img_shape = (channels, h_size, w_size)

        torch.cuda.empty_cache()
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)

        self.data = None

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_epoch = n_epoch
        self.latent_dim = latent_dim

        # Models
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(img_shape)

        # loss
        self.loss = nn.BCELoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        if cuda_C:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss.cuda()

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

    def fit(self, data: DataLoader):
        self.data = data
        for epoch in range(self.n_epoch):
            self.train(epoch)

    def train(self, epoch):

        for step, mat in tqdm(enumerate(self.data)):
            total_step = len(self.data) * epoch + step

            tb = SummaryWriter()

            Tensor = torch.cuda.FloatTensor if cuda_C else torch.FloatTensor

            valid = Variable(Tensor(mat.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.size(0), 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(mat.type(Tensor))
            noise = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], self.latent_dim))))

            # -----------------
            #  Train Generator
            # -----------------
            self.generator_optimizer.zero_grad()

            generated_data = self.generator(noise)
            generator_discriminator_out = self.discriminator(generated_data)

            grid = torchvision.utils.make_grid(generated_data[:4])
            tb.add_image("images", grid, step)
            tb.close()

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

            true_discriminator_out = self.discriminator(real_imgs)
            true_discriminator_loss = self.loss(true_discriminator_out, valid)

            generator_discriminator_out = self.discriminator(generated_data.detach())
            generator_discriminator_loss = self.loss(generator_discriminator_out, fake)

            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if total_step % 10 == 0:
                tb.add_scalar("Loss", generator_discriminator_loss, total_step)
                tb.add_scalar("Correct", self.false_positive(generator_discriminator_out), total_step)
                tb.add_scalar("Accuracy", self.false_positive(generator_discriminator_out) / len(valid), total_step)

    def false_positive(self, out_tensor):
        mask = out_tensor.ge(0.5)
        mask = mask.view(len(out_tensor))
        binary_tensor = torch.cuda.FloatTensor(len(out_tensor)).fill_(0)
        binary_tensor.masked_fill_(mask, 1.)
        return binary_tensor.sum()

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            Tensor = torch.cuda.FloatTensor if cuda_C else torch.FloatTensor
            noise = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            generated_data = self.generator(noise)
            return generated_data
