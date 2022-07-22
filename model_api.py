import torch
from torch import nn
from eng2gloss_dataset import EnglishToGlossDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unknown_idx, padding_idx, bos_idx, eos_idx = 0, 1, 2, 3


def get_square_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def get_masks(eng, gloss):
    eng_seq_len = eng.shape[0]
    gloss_seq_len = gloss.shape[0]

    gloss_mask = get_square_mask(gloss_seq_len)
    eng_mask = torch.zeros((eng_seq_len, eng_seq_len),
                           device=device).type(torch.bool)

    eng_padding_mask = (eng == padding_idx).transpose(0, 1)
    gloss_padding_mask = (gloss == padding_idx).transpose(0, 1)

    return eng_mask, gloss_mask, eng_padding_mask, gloss_padding_mask


class TransformerTranslationModel(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, embedding_size, seq_length, heads, input_vocab_size, output_vocab_size):
        super(TransformerTranslationModel, self).__init__()

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.input_pos_embedding = nn.Embedding(seq_length, embedding_size)
        self.output_pos_embedding = nn.Embedding(seq_length, embedding_size)

        self.transformer = nn.Transformer(d_model=embedding_size, nhead=heads, num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers, dim_feedforward=seq_length)

        self.generator = nn.Linear(embedding_size, output_vocab_size)

    def forward(self, x, y, x_mask, y_mask, x_padding_mask, y_padding_mask, mem_padding_mask):
        x_embedding = self.input_embedding(x)
        y_embedding = self.output_embedding(y)

        yhat = self.transformer(x_embedding, y_embedding, x_mask, y_mask,
                                None, x_padding_mask, y_padding_mask, mem_padding_mask)
        yhat = self.generator(yhat)

        return yhat

    def encode(self, x, x_mask):
        x_embedding = self.input_embedding(x)
        # x_pos_embeddings = self.input_pos_embedding(torch.arange(x.shape[1], device=device))

        # x_embedding = x_embedding + x_pos_embeddings

        return self.transformer.encoder(x_embedding, x_mask)

    def decode(self, y, memory, y_mask):
        y_embedding = self.output_embedding(y)
        # y_pos_embeddings = self.output_pos_embedding(torch.arange(y.shape[1], device=device))

        # y_embedding = y_embedding + y_pos_embeddings

        return self.transformer.decoder(y_embedding, memory, y_mask)


def load_model():
    dataset = EnglishToGlossDataset('english_gloss.csv')

    # for deep model
    eng_vocab_size = len(dataset.vocab_transform_eng)
    gloss_vocab_size = len(dataset.vocab_transform_gloss)
    embedding_size = 512
    heads = 8
    seq_length = 512
    batch_size = 40
    encoder_layers = 6
    decoder_layers = 6

    model = TransformerTranslationModel(
        encoder_layers, decoder_layers, embedding_size, seq_length, heads, eng_vocab_size, gloss_vocab_size).to(device)
    model.load_state_dict(torch.load('model_deep.pt', map_location=device))

    return model, dataset


def decode_greedy(model, eng, eng_mask, max_seq_len):
    eng = eng.to(device)
    eng_mask = eng_mask.to(device)

    memory = model.encode(eng, eng_mask)
    ys = torch.ones(1, 1).fill_(bos_idx).long().to(device)

    for i in range(max_seq_len - 1):
        memory = memory.to(device)
        gloss_mask = (get_square_mask(ys.size(0)).bool()).to(device)

        yhat = model.decode(ys, memory, gloss_mask)
        yhat = yhat.transpose(0, 1)
        probs = model.generator(yhat[:, -1])

        _, next_word = torch.max(probs, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            eng.data).fill_(next_word).to(device)], dim=0)

        if next_word == eos_idx:
            break

    return ys


def translate(model, dataset, eng_sentence):
    eng = dataset.tensor_transform(dataset.vocab_transform_eng(
        dataset.tokenizer(eng_sentence))).view(-1, 1)
    tokens_size = eng.shape[0]

    eng_mask = (torch.zeros(tokens_size, tokens_size)).bool()

    gloss_tokens = decode_greedy(
        model, eng, eng_mask, tokens_size + 5).flatten()

    return (" ".join(dataset.vocab_transform_gloss.lookup_tokens(list(gloss_tokens.cpu().numpy())))).replace('<bos> ', '').replace(' <eos>', '').upper()
