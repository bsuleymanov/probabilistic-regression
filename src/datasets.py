from torch import utils
import pandas as pd
import numpy as np


class PurchasesDataset(utils.data.Dataset):
    def __init__(self, x_filename, y_filename,
                 categorical_features=('chain', 'dept', 'category',
                                       'brand', 'productmeasure'),
                 numeric_features=('numeric',)):
        self._categorical_features = list(categorical_features)
        self._numeric_features = list(numeric_features)
        self._n_categorical = len(self._categorical_features)
        self._n_numeric = len(self._numeric_features)
        self.data = pd.read_csv(x_filename)[
            self.categorical_features + self.numeric_features
        ].astype(np.float32)#.iloc[:1050]
        column_types = {}
        for categorical_column in self.categorical_features:
            column_types[categorical_column] = "long"
        self.data = self.data.astype(column_types)
        self.labels = pd.read_csv(y_filename).values.reshape(-1, 1).astype(np.float32)#[:1050]
        self.labels = np.tile(self.labels, 3)
        #print(self.labels.shape)
        #input()
        self.initialize_vocab_sizes_from_data()
        self.data_df = self.data
        self.data = self.data_df.values

    @property
    def categorical_features(self):
        return self._categorical_features

    @categorical_features.setter
    def categorical_features(self, categorical_features):
        self._categorical_features = categorical_features
        self._n_categorical = len(self._categorical_features)

    @property
    def numeric_features(self):
        return self._numeric_features

    @numeric_features.setter
    def numeric_features(self, numeric_features):
        self._numeric_features = numeric_features
        self._n_numeric = len(self._numeric_features)

    @property
    def n_categorical(self):
        return self._n_categorical

    @property
    def n_numeric(self):
        return self._n_numeric

    @property
    def vocab_sizes(self):
        return self._vocab_sizes

    @vocab_sizes.setter
    def vocab_sizes(self, vocab_sizes):
        if len(self.categorical_features) != len(vocab_sizes):
            raise ValueError(f"Provided vocab_sizes length ({len(vocab_sizes)})"
                             f"doesn't match number of categorical features "
                             f"({self.n_categorical}).")
        for i, new_vocab_size in enumerate(vocab_sizes):
            self._vocab_sizes[i] = max(self._vocab_sizes[i], new_vocab_size)

    def initialize_vocab_sizes_from_data(self):
        data_categorical = self.data[self.categorical_features]
        self._vocab_sizes = [
            int(data_categorical[cat_feature].max()) + 1 for cat_feature in self.categorical_features
        ]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)