import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

##Class defining type of blocks
class Block(nn.Module):
    def __init__(self, channels, mult, block_type, sampling, noise_dim=None, alph=0.01):
        super().__init__()

        if block_type=="g" and sampling == 1:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult + noise_dim + 1, channels*(mult-2), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult-2)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels*(mult-2), channels*(mult-4), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult-4)),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult-4), channels*(mult-6), 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*(mult-6)),
                nn.LeakyReLU(alph),

                nn.ConvTranspose1d(channels*(mult-6), channels*(mult-8), 3, stride = 2, padding=0, bias=False),
                nn.BatchNorm1d(channels*(mult-8)),
                nn.LeakyReLU(alph))

        elif block_type=="d" and sampling == -1:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult + 1, channels*(mult+2), 3, stride = 1, padding=1),
                nn.InstanceNorm1d(channels*(mult+2), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+2), channels*(mult+4), 3, stride = 2, padding=0),
                nn.InstanceNorm1d(channels*(mult+4), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+4), channels*(mult+6), 3, stride = 1, padding=1),
                nn.InstanceNorm1d(channels*(mult+6), affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*(mult+6), channels*(mult+8), 3, stride = 2, padding=0),
                nn.InstanceNorm1d(channels*(mult+8), affine=True),
                nn.LeakyReLU(alph))

        elif block_type=="d" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1),
                nn.InstanceNorm1d(channels*mult, affine=True),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1),
                nn.InstanceNorm1d(channels*mult, affine=True),
                nn.LeakyReLU(alph))

        elif block_type=="g" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph),

                nn.Conv1d(channels*mult, channels*mult, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm1d(channels*mult),
                nn.LeakyReLU(alph))

    def forward(self, x):
        return self.block(x)



class ConvGenerator(nn.Module):
    def __init__(self, latent_size, data_shape, gpu, device, channels, noise_dim, alph):
        super(ConvGenerator, self).__init__()

        #parameters initialization
        self.channels = channels
        self.latent_size = latent_size
        self.alph = alph
        self.gpu = gpu
        self.data_shape = data_shape
        self.device = device
        self.noise_dim = noise_dim

        #Location-specific trainable variables
        self.ms_vars = nn.ParameterList()
        self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size))))
        for i in range(2,15,2):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size*(2**i)-1))))

        #Blocks
        self.block1 = nn.Sequential(
                    nn.Conv1d(noise_dim + 1, self.channels*52, 3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*52),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels*52, self.channels*50, 3, stride = 2, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*50),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels*50, self.channels*48, 3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*48),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels*48, self.channels*46, 3, stride = 2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*46),
                    nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels = self.channels, mult = 46, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block3 = Block(channels = self.channels, mult = 38, block_type = "g", sampling = 0, noise_dim = self.noise_dim)
        self.block4 = Block(channels = self.channels, mult = 38, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block5 = Block(channels = self.channels, mult = 30, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block6 = Block(channels = self.channels, mult = 22, block_type = "g", sampling = 0, noise_dim = self.noise_dim)
        self.block7 = Block(channels = self.channels, mult = 22, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block8 = Block(channels = self.channels, mult = 14, block_type = "g", sampling = 1, noise_dim = self.noise_dim)
        self.block9 = Block(channels = self.channels, mult = 6, block_type = "g", sampling = 0, noise_dim = self.noise_dim)
        self.block10 = nn.Sequential(
                    nn.Conv1d(self.channels * 6 + noise_dim + 1, self.channels * 4, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*4),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels * 4, self.channels * 2, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d(self.channels*2),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 2, self.channels * 1, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(self.channels*1),
                    nn.LeakyReLU(alph),

                    nn.ConvTranspose1d(self.channels * 1, (self.channels * 1)//2, 3, stride=2, padding=0, bias=False),
                    nn.BatchNorm1d((self.channels * 1)//2),
                    nn.LeakyReLU(alph),
        )
        self.block11 = nn.Sequential(
                    nn.Conv1d((self.channels * 1)//2 + 1, 1, 3, stride=1, padding=1),
                    nn.Sigmoid()
        )

    def forward(self, x, noise_list):
        batch_size = x.shape[0]
        x = torch.cat((self.ms_vars[0].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block1(x)

        x = torch.cat((self.ms_vars[1].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[0], x), 1)
        x = self.block2(x)

        res = x
        x = self.block3(x)
        x += res

        x = torch.cat((self.ms_vars[2].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[1], x), 1)
        x = self.block4(x)

        x = torch.cat((self.ms_vars[3].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[2], x), 1)
        x = self.block5(x)

        res = x
        x = self.block6(x)
        x += res

        x = torch.cat((self.ms_vars[4].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[3], x), 1)
        x = self.block7(x)

        x = torch.cat((self.ms_vars[5].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[4], x), 1)
        x = self.block8(x)

        res = x
        x = self.block9(x)
        x += res

        x = torch.cat((self.ms_vars[6].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = torch.cat((noise_list[5], x), 1)
        x = self.block10(x)

        x = torch.cat((self.ms_vars[7].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block11(x)

        return x


class ConvDiscriminator(nn.Module):
    def __init__(self, data_shape, latent_size, gpu, device, pack_m, channels, alph):
        super(ConvDiscriminator, self).__init__()

        #parameters initialization
        self.alph = alph
        self.data_shape = data_shape
        self.channels = channels
        self.gpu = gpu
        self.device = device
        self.pack_m = pack_m

        #Location-specific trainable variables
        self.ms_vars = nn.ParameterList()
        for i in range(14,1,-2):
            self.ms_vars.append(nn.Parameter(torch.normal(mean=0, std=1, size=(1, latent_size*(2**i)-1))))

        #Blocks
        self.block1 = nn.Sequential(
                    nn.Conv1d(1 * pack_m + 1, self.channels * 1, 3, stride=1, padding=1),
                    nn.InstanceNorm1d(self.channels*1, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 1, self.channels * 2, 3, stride=2, padding=0),
                    nn.InstanceNorm1d(self.channels*2, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 2, self.channels * 4, 3, stride=1, padding=1),
                    nn.InstanceNorm1d(self.channels*4, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 4, self.channels * 6, 3, stride=2, padding=0),
                    nn.InstanceNorm1d(self.channels*6, affine=True),
                    nn.LeakyReLU(alph),
        )
        self.block2 = Block(channels = self.channels, mult = 6, block_type = "d", sampling = 0)
        self.block3 = Block(channels = self.channels, mult = 6, block_type = "d", sampling = -1)
        self.block4 = Block(channels = self.channels, mult = 14, block_type = "d", sampling = -1)
        self.block5 = Block(channels = self.channels, mult = 22, block_type = "d", sampling = 0)
        self.block6 = Block(channels = self.channels, mult = 22, block_type = "d", sampling = -1)
        self.block7 = Block(channels = self.channels, mult = 30, block_type = "d", sampling = -1)
        self.block8 = Block(channels = self.channels, mult = 38, block_type = "d", sampling = 0)
        self.block9 = Block(channels = self.channels, mult = 38, block_type = "d", sampling = -1)
        self.block10 = nn.Sequential(
                    nn.Conv1d(self.channels * 46 + 1, self.channels * 48, 3, stride=1, padding=1),
                    nn.InstanceNorm1d(self.channels * 48, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 48, self.channels * 50, 3, stride=2, padding=0),
                    nn.InstanceNorm1d(self.channels*50, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 50, self.channels * 52, 3, stride=1, padding=1),
                    nn.InstanceNorm1d(self.channels * 52, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Conv1d(self.channels * 52, 1, 3, stride=2, padding=1),
                    nn.InstanceNorm1d(1, affine=True),
                    nn.LeakyReLU(alph),

                    nn.Linear(latent_size, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.cat((self.ms_vars[0].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block1(x)

        res = x
        x = self.block2(x)
        x += res

        x = torch.cat((self.ms_vars[1].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block3(x)

        x = torch.cat((self.ms_vars[2].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block4(x)

        res = x
        x = self.block5(x)
        x += res

        x = torch.cat((self.ms_vars[3].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block6(x)

        x = torch.cat((self.ms_vars[4].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block7(x)

        res = x
        x = self.block8(x)
        x += res

        x = torch.cat((self.ms_vars[5].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block9(x)

        x = torch.cat((self.ms_vars[6].repeat(batch_size,1)[:,np.newaxis,:], x), 1)
        x = self.block10(x)

        return x

def gradient_penalty(netC, X_real_batch, X_fake_batch, device):
    batch_size, nb_snps= X_real_batch.shape[0], X_real_batch.shape[2]
    alpha = torch.rand(batch_size,1, device=device).repeat(1, nb_snps)
    alpha = alpha.reshape(alpha.shape[0], 1, alpha.shape[1])
    interpolation = (alpha*X_real_batch) + (1-alpha) * X_fake_batch
    interpolation = interpolation.float()

    interpolated_score= netC(interpolation)

    gradient= torch.autograd.grad(inputs=interpolation,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)

    gradient_penalty *= 10
    return gradient_penalty
