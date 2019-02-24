import torch
import torch.nn as nn
import torch.nn.functional as F


class NonParallelAE(nn.Module):

    def __init__(self, _lambda, _gamma, _drop):
        super(NonParallelAE, self).__init__()
        self.lmbda = _lambda
        self.gamma = _gamma
        self.drop_prob = _drop
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self):
        encode = Encoder()
        generate = Generator()
        align = Aligner()
        discriminate = Discriminator()


class Encoder(nn.Module):
    """
    Get the latent content representations z_k = E(s_k, b_k).
    The encoder is implemented as a single-layer RNN with GRU cell.
    """
    def __init__(self, _s_in, _b_in, _z_dim, _batch_sz, _n_layers, _drop, _embedding_layer):
        self.title_dim = _s_in
        self.bias_dim = _b_in
        self.z_dimension = _z_dim
        self.batch_size = _batch_sz
        self.n_layers = _n_layers
        self.dropout = _drop
        self.embedding = _embedding_layer
        self.encoder_cell = nn.GRUCell(self.z_dimension, self.n_layers, self.dropout)
        self.init_state = None

    def forward(self, title_in, bias_in):
        title_embed = self.embedding(title_in)
        encoder_cell = self.encoder_cell
        # Need  to feed in bias labels as initial state
        z_out = nn.RNN(encoder_cell, title_embed)
        z = z_out[:, bias_in:]
        # Two output states, original latent and transfer latent
        # due to use of Professor-Forcing
        latent_orig = torch.cat([nn.Linear(bias_in, self.bias_dim), z], 1)
        latent_transfer = torch.cat([nn.Lineat(1 - bias_in, self.bias_dim), z], 1)
        return latent_orig, latent_transfer


class Generator(nn.Module):
    """
    Take the latent output from the encoder and generate a sentence s_k conditioned on (z_k, b_k).
    The generator is implemented as a single-layer RNN with GRU cell.
    """
    def __init__(self, _n_layers, _drop, _embedding_layer):
        self.n_layers = _n_layers
        self.dropout = _drop
        self.generator_cell = nn.GRUCell(self.z_dimension, self.n_layers, self.dropout)
        self.embedding = _embedding_layer
        self.init_state = None

    def forward(self, latent_orig_in, latent_transfer_in, bias_in):
        latent_embed = self.embedding(latent_orig_in)
        generator_cell = self.generator_cell
        # Need to feed in bias labels as initial state
        s_hat = nn.RNN(generator_cell, latent_embed, initial_state=self.initial_state)
        teach_s = torch.cat([latent_orig_in, s_hat], 1)
        return s_hat, teach_s


class Aligner(nn.Module):
    def __init__(self):
        pass

    def word_softmax(self):
        pass


class Discriminator(nn.Module):
    def __init__(self, inp_real, inp_fake, filter_sizes, dropout, leak):
        super(Discriminator, self).__init__()
        self.in_real = inp_real
        self.in_fake = inp_fake
        self.out_shape = None
        self.filters = filter_sizes
        self.drop = dropout
        self.leak = leak

        def create_blocks(_in_shape, _out_shape, _leak, _drop):
            conv = nn.Conv2d(_in_shape, _out_shape)
            relu = nn.LeakyReLU(_leak, inplace=True)
            drop = nn.Dropout2d(_drop)
            batch = nn.BatchNorm2d(_out_shape)
            block = [conv, relu, drop, batch]
            return block

        self.model = nn.Sequential(*create_blocks(self.in_real, self.out_shape,
                                                  self.leak, self.drop))
        self.final_layers = nn.Sequential(nn.Linear(), nn.Sigmoid())

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.shape[0], -1)
        discriminator_score = self.final_layers(out)
        return discriminator_score


