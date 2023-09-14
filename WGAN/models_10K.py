import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


##Class defining type of blocks
class Block(nn.Module):
    def __init__(self, channels, mult, block_type, sampling, noise_dim=None, alph=0.01):
        super().__init__()

        if block_type == "g" and sampling == 1:
            self.block = nn.Sequential(
                nn.Conv1d(channels * mult + noise_dim + 1, channels * (mult - 2), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * (mult - 2)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels * (mult - 2), channels * (mult - 4), 3, stride=2, padding=0, bias=False),
                nn.BatchNorm1d(channels * (mult - 4)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * (mult - 4), channels * (mult - 6), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * (mult - 6)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels * (mult - 6), channels * (mult - 8), 3, stride=2, padding=0, bias=False),
                nn.BatchNorm1d(channels * (mult - 8)),
                nn.LeakyReLU(alph))

        elif block_type == "d" and sampling == -1:

            self.block = nn.Sequential(
                nn.Conv1d(channels * mult + 1, channels * (mult + 2), 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * (mult + 2), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * (mult + 2), channels * (mult + 4), 3, stride=2, padding=0),
                nn.InstanceNorm1d(channels * (mult + 4), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * (mult + 4), channels * (mult + 6), 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * (mult + 6), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * (mult + 6), channels * (mult + 8), 3, stride=2, padding=0),
                nn.InstanceNorm1d(channels * (mult + 8), affine=True),
                nn.LeakyReLU(alph))

        elif block_type == "d" and sampling == 0:

            self.block = nn.Sequential(
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * mult, affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * mult, affine=True),
                nn.LeakyReLU(alph))

        elif block_type == "g" and sampling == 0:

            self.block = nn.Sequential(
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * mult),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * mult),
                nn.LeakyReLU(alph))

    def forward(self, x):
        return self.block(x)


import torch
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, latent_size, data_shape, gpu, device, channels, noise_dim, alph, num_classes):
        super(ConvGenerator, self).__init__()

        # parameters initialization
        self.channels = channels
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.ms_vars = nn.ParameterList()

        self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size+ 30))))
        # Location-specific trainable variables
        for i in range(2,14,2):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, (latent_size + 30)*(2**i)-1))))


        # Blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(noise_dim + 1, self.channels * 44, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.channels * 44),
            nn.LeakyReLU(alph),

            nn.ConvTranspose1d(self.channels * 44, self.channels * 42, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.channels * 42),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 42, self.channels * 40, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.channels * 40),
            nn.LeakyReLU(alph),

            nn.ConvTranspose1d(self.channels * 40, self.channels * 38, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(self.channels * 38),
            nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels=self.channels, mult=38, block_type="g", sampling=1, noise_dim=self.noise_dim)
        self.block3 = Block(channels=self.channels, mult=30, block_type="g", sampling=0, noise_dim=self.noise_dim)
        self.block4 = Block(channels=self.channels, mult=30, block_type="g", sampling=1, noise_dim=self.noise_dim)
        self.block5 = Block(channels=self.channels, mult=22, block_type="g", sampling=1, noise_dim=self.noise_dim)
        self.block6 = Block(channels=self.channels, mult=14, block_type="g", sampling=0, noise_dim=self.noise_dim)
        self.block7 = Block(channels=self.channels, mult=14, block_type="g", sampling=1, noise_dim=self.noise_dim)
        self.block8 = nn.Sequential(
            nn.Conv1d(self.channels * 6 + self.noise_dim + 1, self.channels * 4, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(self.channels * 4),
            nn.LeakyReLU(self.alph),

            nn.ConvTranspose1d(self.channels * 4, self.channels * 2, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(self.channels * 2),
            nn.LeakyReLU(self.alph),

            nn.Conv1d(self.channels * 2, self.channels * 1, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(self.channels * 1),
            nn.LeakyReLU(self.alph),

            nn.ConvTranspose1d(self.channels * 1, 1, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(self.alph),
        )


    def forward(self, x, labels, noise_list):
        batch_size = x.shape[0]
        new_embedding = torch.zeros((batch_size, 2, labels.shape[1]))

        for i in range(batch_size):
            new_embedding[i, 0, :] = labels[i, :]
            new_embedding[i, 1, :] = labels[i, :]

        x = torch.cat([new_embedding, x], dim=2)
        x = torch.cat((self.ms_vars[0].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block1(x)

        x = torch.cat((self.ms_vars[1].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = torch.cat((noise_list[0], x), 1)
        x = self.block2(x)

        res = x
        x = self.block3(x)
        x += res

        x = torch.cat((self.ms_vars[2].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = torch.cat((noise_list[1], x), 1)
        x = self.block4(x)

        x = torch.cat((self.ms_vars[3].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = torch.cat((noise_list[2], x), 1)
        x = self.block5(x)

        res = x
        x = self.block6(x)
        x += res

        x = torch.cat((self.ms_vars[4].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = torch.cat((noise_list[3], x), 1)
        x = self.block7(x)

        x = torch.cat((self.ms_vars[5].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = torch.cat((noise_list[4], x), 1)
        x = self.block8(x)
        x = x[:, :, :10000]

        return x


class ConvDiscriminator(nn.Module):
    def __init__(self, data_shape, latent_size, gpu, device, pack_m, channels, alph):
        super(ConvDiscriminator, self).__init__()

        # parameters initialization
        self.alph = alph
        self.data_shape = data_shape
        self.channels = channels
        self.gpu = gpu
        self.device = device
        self.pack_m = pack_m

        # Location-specific trainable variables
        self.ms_vars = nn.ParameterList()
        for i in reversed([2, 8, 38, 155, 624, 2499, 10000]):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, i))))

        # Blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(1 * pack_m + 1, self.channels * 1, 3, stride=1, padding=1),
            nn.InstanceNorm1d(self.channels * 1, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 1, self.channels * 2, 3, stride=2, padding=0),
            nn.InstanceNorm1d(self.channels * 2, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 2, self.channels * 4, 3, stride=1, padding=1),
            nn.InstanceNorm1d(self.channels * 4, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 4, self.channels * 6, 3, stride=2, padding=0),
            nn.InstanceNorm1d(self.channels * 6, affine=True),
            nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels=self.channels, mult=6, block_type="d", sampling=0)
        self.block3 = Block(channels=self.channels, mult=6, block_type="d", sampling=-1)
        self.block4 = Block(channels=self.channels, mult=14, block_type="d", sampling=-1)
        self.block5 = Block(channels=self.channels, mult=22, block_type="d", sampling=0)
        self.block6 = Block(channels=self.channels, mult=22, block_type="d", sampling=-1)
        self.block7 = Block(channels=self.channels, mult=30, block_type="d", sampling=-1)
        self.block8 = nn.Sequential(
            nn.Conv1d(self.channels * 38 + 1, self.channels * 40, 3, stride=1, padding=1),
            nn.InstanceNorm1d(self.channels * 40, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 40, self.channels * 42, 3, stride=2, padding=0),
            nn.InstanceNorm1d(self.channels * 42, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 42, self.channels * 44, 3, stride=1, padding=1),
            nn.InstanceNorm1d(self.channels * 44, affine=True),
            nn.LeakyReLU(alph),

            nn.Conv1d(self.channels * 44, 1, 3, stride=2, padding=1),
            nn.InstanceNorm1d(1, affine=True),
            nn.LeakyReLU(alph),

            nn.Linear(latent_size, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.cat((self.ms_vars[0].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block1(x)

        res = x
        x = self.block2(x)
        x += res

        x = torch.cat((self.ms_vars[1].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block3(x)

        x = torch.cat((self.ms_vars[2].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block4(x)

        res = x
        x = self.block5(x)
        x += res

        x = torch.cat((self.ms_vars[3].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block6(x)

        x = torch.cat((self.ms_vars[4].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block7(x)

        x = torch.cat((self.ms_vars[5].repeat(batch_size, 1)[:, np.newaxis, :], x), 1)
        x = self.block8(x)
        return x


def gradient_penalty(netC, X_real_batch, X_fake_batch, device):
    batch_size, nb_snps = X_real_batch.shape[0], X_real_batch.shape[2]
    alpha = torch.rand(batch_size, 1, device=device).repeat(1, nb_snps)
    alpha = alpha.reshape(alpha.shape[0], 1, alpha.shape[1])
    interpolation = (alpha * X_real_batch) + (1 - alpha) * X_fake_batch
    interpolation = interpolation.float()

    interpolated_score = netC(interpolation)

    gradient = torch.autograd.grad(inputs=interpolation,
                                   outputs=interpolated_score,
                                   retain_graph=True,
                                   create_graph=True,
                                   grad_outputs=torch.ones_like(interpolated_score)
                                   )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    gradient_penalty *= 10
    return gradient_penalty
