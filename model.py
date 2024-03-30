import math

import torch

import config


class ScaledDotProductAttention(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.scale = 1 / math.sqrt(config.d_k)

    def forward(self, query, key, value, attn_mask=None):
        x = query @ key.transpose(-2, -1)  # matrix dot product, swap dims of value
        x *= self.scale  # prevents small gradients from softmax
        if attn_mask is not None:
            x += attn_mask
        x = torch.nn.functional.softmax(x, dim=-1)
        x = x @ value

        return x


class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linear_out = torch.nn.Linear(in_features=config.d_model, out_features=config.d_model)
        self.sdpa = ScaledDotProductAttention()
        self.linear_projs_in = [[
                torch.nn.Linear(in_features=config.d_model, out_features=config.d_k),  # Q
                torch.nn.Linear(in_features=config.d_model, out_features=config.d_k),  # K
                torch.nn.Linear(in_features=config.d_model, out_features=config.d_v),  # V
            ] 
            for _ in range(config.num_heads)
        ]

    def forward(self, x, attn_mask=None):
        batch_size, num_timesteps = x.shape[:2]

        Q = torch.zeros((batch_size, config.num_heads, num_timesteps, config.d_k))
        K = torch.zeros((batch_size, config.num_heads, num_timesteps, config.d_k))
        V = torch.zeros((batch_size, config.num_heads, num_timesteps, config.d_v))

        for i in range(config.num_heads):
            Q[:, i, ...] = self.linear_projs_in[i][0](x)
            K[:, i, ...] = self.linear_projs_in[i][1](x)
            V[:, i, ...] = self.linear_projs_in[i][2](x)
    
        x = self.sdpa(Q, K, V, attn_mask=attn_mask)  # parallel
        x = x.view(batch_size, num_timesteps, config.d_v * config.num_heads)  # concat from the attn heads
        x = self.linear_out(x)
        x = torch.nn.functional.dropout(x, training=True)

        return x


class PositionWiseFeedForward(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=config.d_model, out_features=config.d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config.d_ff, out_features=config.d_model)
        )

    def forward(self, x):
        x = self.ff(x)
        x = torch.nn.functional.dropout(x, training=True)  # TODO: dropout prob?

        return x


class PositionalEncoding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.pe = torch.zeros((config.max_len, config.d_model))
        
        for i in range(config.max_len):
            for j in range(0, config.d_model, 2):
                div_term = 1 / (10000 ** (j / config.d_model))
                self.pe[i][j] = math.sin(i * div_term)
                self.pe[i][j + 1] = math.cos(i * div_term)

    def forward(self, x):
        x += self.pe[:x.shape[1]]

        return x


class EncoderLayer(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.mha = MultiHeadAttention()
        self.ff = PositionWiseFeedForward()

    def forward(self, x): 
        x_mha = self.mha(x)
        x = torch.nn.functional.layer_norm(x + x_mha, [config.d_model])  # residual connection

        x_ff = self.ff(x)
        x = torch.nn.functional.layer_norm(x + x_ff, [config.d_model])

        return x


class Encoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layers = [EncoderLayer() for _ in range(config.num_layers)]
        self.embedding_layer = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.pe_layer = PositionalEncoding()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.pe_layer(x)
        
        for i, encoder_layer in enumerate(self.layers):
            if i > 0:  # queries, keys and values come from the previous encoder layer
                encoder_layer.mha.linear_projs_in = self.layers[i - 1].mha.linear_projs_in
            
            x = encoder_layer(x)  # previous encoder layer output used as input to next encoder layer

        return zip(*self.layers[-1].mha.linear_projs_in)  # TODO: is K, V from last layer correct?


class DecoderLayer(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.masked_mha = MultiHeadAttention()
        self.mha = MultiHeadAttention()
        self.ff = PositionWiseFeedForward()

    def forward(self, x, attn_mask=None):
        x_mmha = self.masked_mha(x, attn_mask=attn_mask)
        x = torch.nn.functional.layer_norm(x + x_mmha, [config.d_model])

        x_mha = self.mha(x)
        x = torch.nn.functional.layer_norm(x + x_mha, [config.d_model])

        x_ff = self.ff(x)
        x = torch.nn.functional.layer_norm(x + x_ff, [config.d_model])

        return x


class Decoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layers = [DecoderLayer() for _ in range(config.num_layers)]
        self.embedding_layer = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)  # TODO: needs shared with encoder
        self.pe_layer = PositionalEncoding()
        self.linear_out = torch.nn.Linear(in_features=config.d_model, out_features=config.vocab_size)

    def generate_mask(self, length):
        mask = torch.triu(torch.ones((length, length))).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        """
        e.g.length = 3: [
                [0, -inf, -inf],
                [0, 0, -inf],
                [0, 0, 0]
            ]
        """

        return mask

    def forward(self, x, key, value):
        x = self.embedding_layer(x)
        x = self.pe_layer(x)

        # K, V of encoder are used in encoder-decoder attention layers of decoder
        # decoding continues until special symbol is reached indicating decoder has completed output
        # decoder output (linear + softmax) at each step is fed to bottom decoder at next time step
        # the decoder self-attention layer uses masking of future positions before softmax step in attention calculation
        # the encoder-decoder attention layer creates it's queries matrix from the layer below it, K, V come from the encoder stack

        num_timesteps = x.shape[1]
        attn_mask = self.generate_mask(length=num_timesteps)

        for i, decoder_layer in enumerate(self.layers):
            if i > 0:
                query, _, _ = zip(*self.layers[i - 1].mha.linear_projs_in)  # query comes from previous decoder layers
            else:
                query, _, _ = zip(*decoder_layer.mha.linear_projs_in)
            
            query = list(query)
            decoder_layer.mha.linear_projs_in = [[query[j], key[j], value[j]] for j in range(config.num_heads)]  # key, value come from encoder
            x = decoder_layer(x, attn_mask=attn_mask)

        x = self.linear_out(x)
        x = torch.nn.functional.softmax(x, dim=-1)

        return x


class Transformer(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, source_x, target_x):
        _, K, V = self.encoder(source_x)  # keys, values from encoder needed for decoder

        return self.decoder(target_x, list(K), list(V))


def main():
    transformer = Transformer()
    
    batch_size = 4
    source_x = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(batch_size)]).int()
    target_x = source_x.clone()

    output = transformer(source_x, target_x)
    print(output.shape)


if __name__ == '__main__':
    main()
