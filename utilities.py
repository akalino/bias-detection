import random
random.seed(17)
import pandas as pd
from nltk.tokenize import sent_tokenize
from torchtext import data


def sample_pairs(_frame, _n):
    ids = list(set(_frame['story_id'].tolist()))
    sample_ids = random.sample(ids, _n)
    sample_frame = _frame[_frame['story_id'].isin(sample_ids)]
    remaining_frame = pd.merge(_frame, sample_frame, on=['story_id', 'bias', 'title', 'content'],
                               how='outer', indicator=True).query("_merge != 'both'")\
        .drop('_merge', axis=1).reset_index(drop=True)
    return remaining_frame, sample_frame


class EncoderData(data.Dataset):

    def __init__(self, path, max_sent_length, **kwargs):
        self.max_sentence_length = max_sent_length
        self.file_path = path
        self.raw_frame = self.load_data()
        self.sentence_frame = self.structure_sentence_inputs()
        STORYID = data.Field()
        BIAS = data.Field()
        TITLE = data.Field()
        CONTENT = data.Field()
        fields = [('story_id', STORYID),
                  ('bias', BIAS),
                  ('title', TITLE),
                  ('content', CONTENT)]
        examples = []
        for idx, vals in self.sentence_frame.iterrows():
            # TODO: padding tag <PAD>
            # TODO: beginning and end of sentence tags <s>, </s>
            examples.append(data.Example.fromlist([vals['story_id'], vals['bias'],
                                                   vals['title'], vals['content']], fields))
        super(EncoderData, self).__init__(examples, fields, **kwargs)

    def load_data(self):
        _frame = pd.read_csv(self.file_path, error_bad_lines=False)
        _frame['sentences'] = _frame['original_body'].apply(lambda x: sent_tokenize(x))
        _frame = _frame[_frame['bias'].isin(['From the Right', 'From the Left'])]
        self.raw_frame = _frame
        return self.raw_frame

    def structure_sentence_inputs(self):
        _frame = self.raw_frame[['story_id', 'bias', 'title', 'sentences']]
        rows = []
        _ = _frame.apply(lambda row: [rows.append([row['story_id'], row['bias'],
                                                   row['title'], sent])
                                      for sent in row.sentences], axis=1)
        _new_frame = pd.DataFrame(rows)
        _new_frame.columns = ['story_id', 'bias', 'title', 'content']
        self.sentence_frame = _new_frame
        self.sentence_frame = self.recode_bias()
        return self.sentence_frame

    def recode_bias(self):
        self.sentence_frame = self.sentence_frame.replace(['From the Left', 'From the Right'],
                                                          [0, 1])
        return self.sentence_frame


def preprocess(vocab_size, batch_size, max_sent_len, _path):
    train_titles = data.Field()
    train_content = data.Field()
    val_titles = data.Field()
    val_content = data.Field()
    test_titles = data.Field()
    test_content = data.Field()
    loaded_data = EncoderData(_path, max_sent_len)
    if vocab_size > 0:
        titles = train_titles.build_vocab(loaded_data.title, max_size=vocab_size)
        contents = train_content.build_vocab(loaded_data.content, max_size=vocab_size)
    else:
        titles = train_titles.build_vocab(loaded_data.title)
        contents = train_content.build_vocab(loaded_data.content)

    # Create iterators to process text in batches of approx. the same length
    train_iter_pos = data.BucketIterator(train_pos, batch_size=batch_size, device=-1, repeat=True,
                                         sort_key=lambda x: len(x.text))
    train_iter_neg = data.BucketIterator(train_neg, batch_size=batch_size, device=-1, repeat=True,
                                         sort_key=lambda x: len(x.text))
    val_iter_pos = data.BucketIterator(val_pos, batch_size=1, device=-1, repeat=True,
                                       sort_key=lambda x: len(x.text))
    val_iter_neg = data.BucketIterator(val_neg, batch_size=1, device=-1, repeat=True,
                                       sort_key=lambda x: len(x.text))
