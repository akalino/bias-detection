import torch
import torch.nn as nn
import torch.nn.functional as F


class NonParallelAE(nn.Module):

    def __init__(self, _lambda, _gamma, _drop, _vocab_size, _embed_size):
        super(NonParallelAE, self).__init__()
        self.lmbda = _lambda
        self.gamma = _gamma
        self.drop_prob = _drop
        self.embedding_size = _embed_size
        self.vocab_size = _vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

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
    def __init__(self, _s_in, _z_dim, _batch_sz, _n_layers, _drop, _embedding, _embed_size):
        super(Encoder, self).__init__()
        self.s_dim = _s_in
        self.z_dimension = _z_dim
        self.batch_size = _batch_sz
        self.n_layers = _n_layers
        self.dropout = _drop
        self.embedding = _embedding
        self.embedding_size = _embed_size
        self.encoder_cell = nn.GRU(input_size=self.embedding_size, hidden_size=self.z_dimension,
                                   num_layers=self.n_layers, dropout=self.dropout)
        self.init_state = nn.Linear(1, self.z_dimension)
        self.unroller = [torch.zeros(self.batch_size, self.z_dimension)] * (self.n_layers - 1)
        self.recurrent = nn.RNN(input_size=self.z_dimension,
                                hidden_size=self.z_dimension,
                                num_layers=self.n_layers)

    def forward(self, title_in, bias_in):
        title_embed = self.embedding(title_in)
        # unrolled needs to have dim (num_layers*num_directions, batch, hidden_size)
        bias_in = bias_in.reshape([-1, 1])
        stacker_list = [self.init_state(bias_in.float())] + self.unroller
        unrolled = torch.stack(stacker_list, 0)
        cell_out = self.encoder_cell(title_embed)
        z_out, h_out = self.recurrent(cell_out[0], unrolled)
        z = z_out[0, :, :]
        return z


class Generator(nn.Module):
    """
    Take the latent output from the encoder and generate a sentence s_k conditioned on (z_k, b_k).
    The generator is implemented as a single-layer RNN with GRU cell.
    """
    def __init__(self, _n_layers, _drop, _s_dim, _z_dim, _embedding, _vocab_sz, _embed_sz):
        super(Generator, self).__init__()
        self.n_layers = _n_layers
        self.dropout = _drop
        self.s_out_dim = _s_dim
        self.z_dimension = _z_dim
        self.embedding_size = _embed_sz
        self.generator_cell = nn.GRU(input_size=self.embedding_size,
                                     hidden_size=self.z_dimension,
                                     num_layers=self.n_layers,
                                     dropout=self.dropout)
        self.latent_linear = nn.Linear(1, self.z_dimension)
        self.recurrent = nn.RNN(input_size=self.z_dimension,
                                hidden_size=self.z_dimension,
                                num_layers=self.n_layers)
        self.embedding = _embedding
        self.vocab_size = _vocab_sz
        self.drop_layer = nn.Dropout(self.dropout)
        self.vocab_proj = nn.Linear(self.z_dimension, self.vocab_size)
        self.vocab_bias = nn.Parameter(torch.ones(self.vocab_size))

    def forward(self, est_z, act_z, bias_in):
        # Two output states, original latent and transfer latent
        content_embed = self.embedding(act_z)
        bias_in = bias_in.reshape([-1, 1])
        latent_orig = torch.stack([self.latent_linear(bias_in.float()), est_z], 0)
        latent_transfer = torch.stack([self.latent_linear(1 - bias_in.float()), est_z], 0)
        generator_out = self.generator_cell(content_embed)
        s_hat_out, h_hat_out = self.recurrent(generator_out[0], latent_orig)
        teach_s = torch.cat([latent_orig, s_hat_out], 0)  # Need to align params here
        s_hat_drop = self.drop_layer(s_hat_out)
        s_hat_logit = self.vocab_proj(s_hat_drop) + self.vocab_bias
        return s_hat_logit, teach_s, latent_transfer


class Decoder(nn.Module):
    def __init__(self, _vocab_proj, _vocab_bias, _dropout, _embedding, _gamma):
        self.vocab_proj = _vocab_bias
        self.vocab_bias = _vocab_bias
        self.dropout = _dropout
        self.embedding = _embedding
        self.gamma = _gamma
        self.drop_layer = nn.Dropout(self.drop)

    def soft_word_output(self, output):
        """
        Create logits, get gumbel-softmax probabilities and select highest probability words from
        embedding layer. Function applies to first pass only.

        :param output: Prior soft-output.
        :return: Transferred hidden state trsf, logits for future use.
        """
        drop_out = self.drop_layer(output)
        out_logit = torch.mm(self.vocab_proj, drop_out) + self.vocab_bias
        probs = F.gumbel_softmax(out_logit, self.gamma)
        trsf = torch.mm(probs, self.embedding)
        return trsf, out_logit

    def softmax_words(self, output):
        """
        Create logits, get gumbel-softmax probabilities and select highest probability words from
        embedding layer. Function applies only after first pass.

        :param output:
        :return:
        """

    def forward(self):




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


