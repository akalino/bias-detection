from utilities import *
from models.model import Encoder, Generator
import torch.nn as nn

PATH = '/Users/alexanderkalinowski/PycharmProjects/bias-detection/data/allsides-collection/train_bias.csv'
title_size = 256
content_size = 512
vocab_size = 10000
batch_size = 32
n_layers = 2
dropout = 0.5
embedding_size = 100

SI = StoryInputs(PATH, title_size, content_size, vocab_size, batch_size)
bat = next(iter(SI.train_iter))

bias_in = bat.bias
title_in = bat.title
content_in = bat.content

embedding_layer = nn.Embedding(vocab_size, embedding_size)

enc = Encoder(title_size, content_size, batch_size, n_layers, dropout, embedding_layer, embedding_size)

out = enc(title_in, bias_in)
print('Title encoded into content space')
print(out.shape)
 
gen = Generator(n_layers, dropout, title_size, content_size, embedding_layer, vocab_size, embedding_size)

s_logit, teacher, lat = gen(out, content_in, bias_in)

print(s_logit.shape)
print(teacher.shape)
print(lat.shape)

