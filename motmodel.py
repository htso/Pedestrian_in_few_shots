import os
import sys
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
from torch.autograd import Variable

try:
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)
except ModuleNotFoundError:
    # put parent directory in path for utils
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)

# Model
class Statistician(nn.Module):
    def __init__(self, batch_size=16, K=5, n_features=1,
                 height=160, width=96,
                 c_dim=3, n_hidden_statistic=128, hidden_dim_statistic=3,
                 n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                 nonlinearity=F.relu, print_vars=False, debug=False):

        super(Statistician, self).__init__()
        self.debug = debug
        # data shape
        self.batch_size = batch_size
        self.K = K
        self.n_features = n_features

        # context
        self.c_dim = c_dim
        self.n_hidden_statistic = n_hidden_statistic
        self.hidden_dim_statistic = hidden_dim_statistic

        # latent
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        self.height = height
        self.width = width

        # modules
        # convolutional encoder
        self.shared_convolutional_encoder = SharedConvolutionalEncoder(self.nonlinearity)

        # statistic network
        statistic_args = (self.batch_size, self.K, self.n_features,
                          self.n_hidden_statistic, self.hidden_dim_statistic,
                          self.c_dim, self.nonlinearity)
        self.statistic_network = StatisticNetwork(*statistic_args)

        z_args = (self.batch_size, self.K, self.n_features,
                  self.n_hidden, self.hidden_dim, self.c_dim, self.z_dim,
                  self.nonlinearity)
        # inference networks
        self.inference_networks = nn.ModuleList([InferenceNetwork(*z_args)
                                                 for _ in range(self.n_stochastic)])

        # latent decoders
        self.latent_decoders = nn.ModuleList([LatentDecoder(*z_args)
                                              for _ in range(self.n_stochastic)])

        # observation decoder
        observation_args = (self.batch_size, self.K, self.n_features,
                            self.width, self.height, 
                            self.n_hidden, self.hidden_dim, self.c_dim,
                            self.n_stochastic, self.z_dim, self.nonlinearity)
        self.observation_decoder = ObservationDecoder(*observation_args)

        # initialize weights
        self.apply(self.weights_init)

        # print variables for sanity check and debugging
        if print_vars:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()

    def forward(self, x):
        if self.debug is True:
            print('x :', x.shape)
        # x : (5, 2, 3, 160, 96)
        # where 
        # 5 = batch size, 
        # 2 = sample size, 
        # 3 = # channels, 
        # 160 = height, 
        # 96 = width
        # NOTE : each batch may belong to different alphabet (or class), but all images within a batch
        # belong to the same alphabet (class).

        # convolutional encoder
        h = self.shared_convolutional_encoder(x, width=self.width, height=self.height)
        if self.debug is True:
            print('h :', h.shape)
        # h : (10, 256, 10, 6)
        # where 
        #     10 = batch_size x K
        #    256 = # kernels output from share_conv_encoder
        #     10 = 160 / 2^4
        #      6 =  96 / 2^4
        # NOTE : I don't understand how this works here. The first dimension of h suggests
        # the entries in the batch are meshed (entangled) together, even though each row in
        # the batch may belong to different alphabet. 

        # statistic network
        c_mean, c_logvar = self.statistic_network(h)
        
        if self.training:
            c = self.reparameterize_gaussian(c_mean, c_logvar)
        else:  # sampling conditioned on inputs
            c = c_mean

        if self.debug is True:
            print('c :', c.shape)   
        # c : (5, 32)    
        # 5 = batch_size, 
        # 32 = c-dim

        # inference networks
        qz_samples = []
        qz_params = []
        z = None
        for inference_network in self.inference_networks:
            z_mean, z_logvar = inference_network(h, z, c)
            qz_params.append([z_mean, z_logvar])
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            qz_samples.append(z)

        if self.debug is True:
            print('z :', z.shape)        
        # z : (10, 32)
        # 10 = batch_size x K
        # 32 = z-dim

        # latent decoders
        pz_params = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            pz_params.append([z_mean, z_logvar])
            z = qz_samples[i]

        if self.debug is True:
            print('z :', z.shape)       
        # z : (10, 32) 

        zs = torch.cat(qz_samples, dim=1)

        if self.debug is True:
            print('zs :', zs.shape)        
        # zs : (10, 192)
        # 10 = batch_size x K
        # 192 = z-dim x n-stochastic = 32 x 6

        # observation decoder
        x_mean, x_logvar = self.observation_decoder(zs, c)

        if self.debug is True:
            print('x_mean :', x_mean.shape)        
        # x : (10, 3, 160, 96)
        # 10 = K x batch_size
        # 3 = $ channels
        # 160 = height
        # 96 = width

        outputs = (
            (c_mean, c_logvar),
            (qz_params, pz_params),
            (x, x_mean, x_logvar)
        )

        return outputs

    def loss(self, outputs, weight):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mean, x_logvar = x_outputs
        recon_loss = gaussian_log_likelihood(x.view(-1, 3, self.height, self.width), x_mean, x_logvar, clip=True)
        recon_loss /= (self.batch_size * self.K)

        # 2. KL Divergence terms
        kl = 0

        # a) Context divergence
        c_mean, c_logvar = c_outputs
        kl_c = kl_diagnormal_stdnormal(c_mean, c_logvar)
        kl += kl_c

        # b) Latent divergences
        qz_params, pz_params = z_outputs
        shapes = (
            (self.batch_size, self.K, self.z_dim),
            (self.batch_size, 1, self.z_dim)
        )
        for i in range(self.n_stochastic):
            args = (qz_params[i][0].view(shapes[0]),
                    qz_params[i][1].view(shapes[0]),
                    pz_params[i][0].view(shapes[1] if i == 0 else shapes[0]),
                    pz_params[i][1].view(shapes[1] if i == 0 else shapes[0]))
            kl_z = kl_diagnormal_diagnormal(*args)
            kl += kl_z

        kl /= (self.batch_size * self.K)

        # Variational lower bound and weighted loss
        vlb = recon_loss - kl
        loss = - ((weight * recon_loss) - (kl / weight))

        return loss, vlb


    def step(self, inputs, alpha, optimizer, clip_gradients=True):
        assert self.training is True
        if self.debug is True:
            print('step inputs :', inputs.shape)
        outputs = self.forward(inputs)
        loss, vlb = self.loss(outputs, weight=(alpha + 1))

        # perform gradient update
        optimizer.zero_grad()
        loss.backward()
        if clip_gradients:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
        optimizer.step()

        # output variational lower bound for batch
        return vlb.item()

    def sample(self):
        c = torch.randn(self.batch_size, self.c_dim)

        # latent decoders
        pz_samples = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            pz_samples.append(z)

        # observation decoder
        zs = torch.cat(pz_samples, dim=1)
        x_mean, x_logvar = self.observation_decoder(zs, c)

        return x_mean

    def sample_conditioned(self, inputs):
        if self.debug is True:
            print('[sample_conditioned] inputs shpae :', inputs.shape)
        h = self.shared_convolutional_encoder(inputs, width=self.width, height=self.height)
        if self.debug is True:
            print('[sample_conditioned] h shpae :', h.shape)
        c, _ = self.statistic_network(h)

        # latent decoders
        pz_samples = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            if i == 0:
                z_mean = z_mean.repeat(self.K, 1)
                z_logvar = z_logvar.repeat(self.K, 1)
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            pz_samples.append(z)

        # observation decoder
        zs = torch.cat(pz_samples, dim=1)
        x_mean, x_logvar = self.observation_decoder(zs, c)

        return x_mean

    def save(self, optimizer, path):
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, path)

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda())
        return mean + std * eps

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class FCResBlock(nn.Module):
    '''
    Module for residual/skip connections
    '''
    def __init__(self, dim, n, nonlinearity, batch_norm=True):

        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.block = nn.ModuleList(
                [nn.ModuleList([nn.Linear(dim, dim), nn.BatchNorm1d(num_features=dim)])
                 for _ in range(self.n)]
            )
        else:
            self.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n)])

    def forward(self, x):
        if self.debug is True:
            print('[FCResBlock forward] x shape :', x.shape)

        e = x + 0

        if self.batch_norm:
            for i, pair in enumerate(self.block):
                fc, bn = pair
                e = fc(e)
                e = bn(e)
                if i < (self.n - 1):
                    e = self.nonlinearity(e)
        else:
            for i, layer in enumerate(self.block):
                e = layer(e)
                if i < (self.n - 1):
                    e = self.nonlinearity(e)

        if self.debug is True:
            print('[FCResBlock output] e shape :', e.shape)            
        return self.nonlinearity(e + x) # <-- skip connection


class Conv2d3x3(nn.Module):
    '''
    Building block for convolutional encoder with same padding
    '''
    def __init__(self, in_channels, out_channels, downsample=False, debug=False):
        super(Conv2d3x3, self).__init__()
        self.debug = debug
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, stride=stride)
    def forward(self, x):
        return self.conv(x)


class SharedConvolutionalEncoder(nn.Module):
    '''
    SHARED CONVOLUTIONAL ENCODER
    NOTE : it takes color images as input, thus the in_channels=3 in the first conv layer
    '''
    def __init__(self, nonlinearity, debug=False):
        super(SharedConvolutionalEncoder, self).__init__()
        self.debug = debug
        self.nonlinearity = nonlinearity
        # input x : (-1, 3, 160, 96)
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=3, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 80, 48)
            Conv2d3x3(in_channels=32, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64, downsample=True),
            # shape is now (-1, 64, 40, 24)
            Conv2d3x3(in_channels=64, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128, downsample=True),
            # shape is now (-1, 128, 20, 12)
            Conv2d3x3(in_channels=128, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256, downsample=True)
            # shape is now (-1, 256, 10, 6)  ==> n_features = 256*10*6
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
        ])

    def forward(self, x, width, height):
        if self.debug is True:
            print('[SharedConvolutionalEncoder, forward] x shape:', x.shape)
        #h = x.view(-1, 3, 64, 64)
        h = x.view(-1, 3, height, width)
        if self.debug is True:
            print('[SharedConvolutionalEncoder, forward] h shape:', h.shape)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.nonlinearity(h)
        return h


# PRE-POOLING FOR STATISTIC NETWORK
class PrePool(nn.Module):

    def __init__(self, batch_size, n_features, n_hidden, hidden_dim, nonlinearity, debug):
        super(PrePool, self).__init__()
        self.debug = debug
        self.batch_size = batch_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc = nn.Linear(self.n_features, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, h):
        #print('PrePool h :', h.shape)        
        # h : (10, 256, 10, 6)
        # where 
        #     10 = batch_size x K
        #    256 = # kernels output from share_conv_encoder
        #     10 = 160 / 2^4
        #      6 =  96 / 2^4
        # reshape and affine
        e = h.view(-1, self.n_features)
        #print('PrePool e :', e.shape)        
        e = self.fc(e)
        #print('PrePool e :', e.shape)
        e = self.bn(e)
        #print('PrePool e :', e.shape)
        e = self.nonlinearity(e)
        #print('PrePool e :', e.shape)
        return e


# POST POOLING FOR STATISTIC NETWORK
class PostPool(nn.Module):

    def __init__(self, n_hidden, hidden_dim, c_dim, nonlinearity, debug=False):
        super(PostPool, self).__init__()
        self.debug = debug
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.Linear(self.hidden_dim, self.hidden_dim)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim)])

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            e = fc(e)
            e = bn(e)
            e = self.nonlinearity(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.c_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.c_dim)

        mean, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mean, logvar


# STATISTIC NETWORK q(c|D)
class StatisticNetwork(nn.Module):

    def __init__(self, batch_size, K, n_features,
                 n_hidden, hidden_dim, c_dim, nonlinearity, debug=False):
        super(StatisticNetwork, self).__init__()
        self.debug = debug
        self.batch_size = batch_size
        self.K = K
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.prepool = PrePool(self.batch_size, self.n_features,
                               self.n_hidden, self.hidden_dim, self.nonlinearity)
        self.postpool = PostPool(self.n_hidden, self.hidden_dim,
                                 self.c_dim, self.nonlinearity)

    def forward(self, h):
        #print('StatisticNetwork, forward, h:', h.shape)
        e = self.prepool(h)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    # Take average of each batch over the sample. Pooling <==> averaging
    def pool(self, e):
        #print('pool e :', e.shape)        
        e = e.view(self.batch_size, self.K, self.hidden_dim)
        e = e.mean(1).view(self.batch_size, self.hidden_dim)
        return e

# Inference network q(z|h, z, c) gives approximate posterior over latent variables.
class InferenceNetwork(nn.Module):
  
    def __init__(self, batch_size, K, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity, debug=False):
        super(InferenceNetwork, self).__init__()
        self.debug = debug
        self.batch_size = batch_size
        self.K = K
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_h = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       nonlinearity=self.nonlinearity, batch_norm=True)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, h, z, c):
        # combine h, z, and c
        # embed h
        eh = h.view(-1, self.n_features)
        eh = self.fc_h(eh)
        eh = eh.view(self.batch_size, self.K, self.hidden_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.K, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(eh.size()).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(eh)

        # sum and reshape
        e = eh + ez + ec
        e = e.view(self.batch_size * self.K, self.hidden_dim)
        e = self.nonlinearity(e)

        # for layer in self.fc_block:
        e = self.fc_res_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# LATENT DECODER p(z|z, c)
class LatentDecoder(nn.Module):
 
    def __init__(self, batch_size, K, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(LatentDecoder, self).__init__()
        self.batch_size = batch_size
        self.K = K
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       nonlinearity=self.nonlinearity, batch_norm=True)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, z, c):
        # combine z and c
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.K, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(self.batch_size, 1, self.hidden_dim).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ez)

        # sum and reshape
        e = ez + ec
        e = e.view(-1, self.hidden_dim)
        e = self.nonlinearity(e)

        # for layer in self.fc_block:
        e = self.fc_res_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# Observation Decoder p(x|z, c)
class ObservationDecoder(nn.Module):
   
    def __init__(self, batch_size, K,  n_features, width, height,
                 n_hidden, hidden_dim, c_dim, n_stochastic, z_dim,
                 nonlinearity):
        super(ObservationDecoder, self).__init__()
        self.batch_size = batch_size
        self.K = K
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.n_stochastic = n_stochastic
        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        self.height = height
        self.width = width

        # shared learnable log variance parameter
        self.logvar = nn.Parameter(torch.randn(1, 3, self.height, self.width).cuda())

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_initial = nn.Linear(self.hidden_dim, self.hidden_dim)
        # ----extra -------------------------------------------------
        self.fc_linear = nn.Linear(self.hidden_dim, self.n_features)

        # Total of 12 layers : 8 conv, 4 transpose
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=256, out_channels=256), # C1
            Conv2d3x3(in_channels=256, out_channels=256), # C2
            nn.ConvTranspose2d(in_channels=256, out_channels=256,
                               kernel_size=2, stride=2),  # T1 
            Conv2d3x3(in_channels=256, out_channels=128), # C3
            Conv2d3x3(in_channels=128, out_channels=128), # C4
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=2, stride=2),  # T2
            Conv2d3x3(in_channels=128, out_channels=64),  # C5
            Conv2d3x3(in_channels=64, out_channels=64),   # C6
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=2, stride=2),  # T3
            Conv2d3x3(in_channels=64, out_channels=32),   # C7
            Conv2d3x3(in_channels=32, out_channels=32),   # C8
            nn.ConvTranspose2d(in_channels=32, out_channels=32,
                               kernel_size=2, stride=2)   # T4
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
        ])

        self.conv_mean = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, zs, c):
        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.K, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)

        e = ezs + ec
        e = self.nonlinearity(e)
        e = e.view(-1, self.hidden_dim)

        e = self.fc_initial(e)
        e = self.nonlinearity(e)
        e = self.fc_linear(e)
        # =========================
        e = e.view(-1, 256, 10, 6)
        # =========================

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = self.nonlinearity(e)

        mean = self.conv_mean(e)
        mean = torch.sigmoid(mean) # previously, F.sigmoid(mean)

        return mean, self.logvar.expand_as(mean)
        