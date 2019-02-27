import random
random.seed(17)
import pandas as pd
from nltk.tokenize import sent_tokenize
import torch
from torchtext import data


def process_raw_data(_path, _train_sz, _val_sz, _test_sz):
    _frame = pd.read_csv(_path, error_bad_lines=False)
    _frame['sentences'] = _frame['original_body'].apply(lambda x: sent_tokenize(x))
    _frame = _frame[_frame['bias'].isin(['From the Right', 'From the Left'])]
    _frame = _frame[['story_id', 'bias', 'title', 'sentences']]
    rows = []
    _ = _frame.apply(lambda row: [rows.append([row['story_id'], row['bias'],
                                               row['title'], sent])
                                  for sent in row.sentences], axis=1)
    _new_frame = pd.DataFrame(rows)
    _new_frame.columns = ['story_id', 'bias', 'title', 'content']
    _new_frame = _new_frame.replace(['From the Left', 'From the Right'], [0, 1])
    _all_ids = list(set(_new_frame.story_id.tolist()))
    train_sample_size = int(len(_all_ids) * _train_sz)
    val_sample_size = int(len(_all_ids) * _val_sz)
    train_ids = _all_ids[:train_sample_size]
    val_ids = _all_ids[train_sample_size+1:train_sample_size+val_sample_size]
    test_ids = _all_ids[train_sample_size+val_sample_size+1:]
    train_frame = _new_frame[_new_frame.story_id.isin(train_ids)]
    valid_frame = _new_frame[_new_frame.story_id.isin(val_ids)]
    test_frame = _new_frame[_new_frame.story_id.isin(test_ids)]
    return train_frame, valid_frame, test_frame


class StoryInputs(data.Dataset):
    def __init__(self, _path, _title_max, _content_max, _vocab_sz, _batch_size):
        self.path = _path
        self.title_len = _title_max  # dim y
        self.content_len = _content_max  # dim z
        self.vocab_size = _vocab_sz
        self.batch_size = _batch_size
        splitter = lambda x: x.split()
        STORY_ID = data.Field(sequential=False, use_vocab=False)
        BIAS = data.Field(sequential=False, use_vocab=False, is_target=True)
        TITLE = data.Field(sequential=True, tokenize=splitter, lower=True,
                           init_token="<t>", eos_token="</t>", fix_length=self.title_len)
        CONTENT = data.Field(sequential=True, tokenize=splitter, lower=True,
                             init_token="<s>", eos_token="</s>", fix_length=self.content_len)
        news_data = data.TabularDataset(path=self.path, format='csv',
                                        fields=[('story_id', STORY_ID),
                                                ('bias', BIAS),
                                                ('title', TITLE),
                                                ('content', CONTENT)])
        TITLE.build_vocab(news_data.title, news_data.content, max_size=self.vocab_size)
        CONTENT.vocab = TITLE.vocab
        # Need to adjust the batching to get equal samples of stories with swapped bias
        self.train_iter = data.BucketIterator(dataset=news_data, batch_size=self.batch_size,
                                              sort_key=lambda x: data.interleave_keys(len(x.TITLE),
                                                                                      len(x.CONTENT)))
