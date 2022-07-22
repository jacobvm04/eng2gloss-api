import torch
import string
import torchtext
import pandas as pd


def pre_process_text(text):
    sent = ""
    for word in text.split():
        use_word = word
        if len(word) >= len("DESC") and word[:len("DESC")] == "DESC":
            use_word = word[len("DESC"):]

        if use_word[0] == "X":
            use_word = use_word[1:]

        if use_word[-2:] == "wh":
            use_word = use_word[:-2]
        sent += use_word + " "

    return str(sent.translate(str.maketrans('', '', string.punctuation)).lower())[:-1]


unknown_idx, padding_idx, bos_idx, eos_idx = 0, 1, 2, 3
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']


class EnglishToGlossDataset(torch.utils.data.Dataset):
    @staticmethod
    def tensor_transform(tokens_idxs):
        return torch.concat((
            torch.tensor([bos_idx]),
            torch.tensor(tokens_idxs),
            torch.tensor([eos_idx])
        )).long()

    @staticmethod
    def collate(batch):
        en_batch, gloss_batch = [], []

        for en, gloss in batch:
            en_batch.append(en)
            gloss_batch.append(gloss)

        en_batch = torch.nn.utils.rnn.pad_sequence(
            en_batch, padding_value=padding_idx)
        gloss_batch = torch.nn.utils.rnn.pad_sequence(
            gloss_batch, padding_value=padding_idx)

        return en_batch, gloss_batch

    def __init__(self, csv_filename):
        self.df = pd.read_csv(csv_filename)
        self.df['en'] = self.df['en'].apply(pre_process_text)
        self.df['gloss'] = self.df['gloss'].apply(pre_process_text)

        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        def yield_tokens_eng():
            for line in self.df['en']:
                yield self.tokenizer(line)

        def yield_tokens_gloss():
            for line in self.df['gloss']:
                yield self.tokenizer(line)

        self.vocab_transform_eng = torchtext.vocab.build_vocab_from_iterator(
            yield_tokens_eng(),
            min_freq=1,
            specials=special_tokens,
            special_first=True,
        )

        self.vocab_transform_gloss = torchtext.vocab.build_vocab_from_iterator(
            yield_tokens_gloss(),
            min_freq=1,
            specials=special_tokens,
            special_first=True,
        )

        self.vocab_transform_eng.set_default_index(unknown_idx)
        self.vocab_transform_gloss.set_default_index(unknown_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        en = self.tensor_transform(
            self.vocab_transform_eng(self.tokenizer(item['en'])))
        gloss = self.tensor_transform(
            self.vocab_transform_gloss(self.tokenizer(item['gloss'])))

        return en, gloss
