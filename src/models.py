import torch
import torch.nn as nn
import numpy as np

from collections.abc import Iterable


class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation="relu"):
        super(DenseLayer, self).__init__()
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid()
        }
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activations[activation]
        )

    def forward(self, x):
        return self.layer(x)

class BaseMLP(nn.Module):
    def __init__(self, n_categorical, n_numerical,
                 vocab_sizes, emb_dims, layer_dims,
                 output_dim, activation="relu"):
        super(BaseMLP, self).__init__()
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical
        if not isinstance(vocab_sizes, Iterable):
            vocab_sizes = [vocab_sizes for x in range(len(n_categorical))]
        else:
            if len(vocab_sizes) != n_categorical:
                raise ValueError(f"Length of vocab_sizes ({len(vocab_sizes)}) "
                                 f"doesn't equal to n_categorical ({n_categorical}).")
        if not isinstance(emb_dims, Iterable):
            emb_dims = [emb_dims for x in range(n_categorical)]
        else:
            if len(emb_dims) != n_categorical:
                raise ValueError(f"Length of emb_dims ({len(emb_dims)}) "
                                 f"doesn't equal to n_categorical ({n_categorical}).")
        self.embedders = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], emb_dims[i])
            for i in range(len(vocab_sizes))
        ])
        input_dim = sum(emb_dims) + n_numerical
        self.layers = [DenseLayer(input_dim, layer_dims[0])]
        for i in range(len(layer_dims[1:-1])):
            self.layers.append(DenseLayer(layer_dims[i-1], layer_dims[i]))
        self.layers.append(DenseLayer(layer_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, data):
        categorical_data = data[:, :self.n_categorical].long()
        embedded_categorical_data = [
            embedder(categorical_data[:, i]) for i, embedder in enumerate(self.embedders)
        ]
        numerical_data = data[:, self.n_categorical:]
        embedded_input = torch.cat(embedded_categorical_data + [numerical_data], dim=1)
        output = self.layers(embedded_input)
        return output

class ZILNMLP(nn.Module):
    def __init__(self, n_categorical, n_numerical,
                 vocab_sizes, emb_dims, layer_dims,
                 output_dim=3, activation="relu",
                 return_logits=True):
        super(ZILNMLP, self).__init__()
        self.return_logits = return_logits
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical
        if not isinstance(vocab_sizes, Iterable):
            vocab_sizes = [vocab_sizes for x in range(len(n_categorical))]
        else:
            if len(vocab_sizes) != n_categorical:
                raise ValueError(f"Length of vocab_sizes ({len(vocab_sizes)}) "
                                 f"doesn't equal to n_categorical ({n_categorical}).")
        if not isinstance(emb_dims, Iterable):
            emb_dims = [emb_dims for x in range(n_categorical)]
        else:
            if len(emb_dims) != n_categorical:
                raise ValueError(f"Length of emb_dims ({len(emb_dims)}) "
                                 f"doesn't equal to n_categorical ({n_categorical}).")
        self.embedders = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], emb_dims[i])
            for i in range(len(vocab_sizes))
        ])
        input_dim = sum(emb_dims) + n_numerical
        self.layers = [DenseLayer(input_dim, layer_dims[0])]
        for i in range(1, len(layer_dims)):
            self.layers.append(DenseLayer(layer_dims[i-1], layer_dims[i]))
        self.layers.append(DenseLayer(layer_dims[-1], output_dim))
        print(self.layers)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, data):
        categorical_data = data[:, :self.n_categorical].long()
        embedded_categorical_data = [
            embedder(categorical_data[:, i]) for i, embedder in enumerate(self.embedders)
        ]
        numerical_data = data[:, self.n_categorical:]
        embedded_input = torch.cat(embedded_categorical_data + [numerical_data], dim=1)
        self.logits = self.layers(embedded_input)
        positive_probs = nn.functional.sigmoid(self.logits[:, :1])
        loc = self.logits[:, 1:2]
        scale = nn.functional.softplus(self.logits[:, 2:])
        return positive_probs * torch.exp(loc + 0.5 * torch.square(scale))


class CatboostModel:
    ...