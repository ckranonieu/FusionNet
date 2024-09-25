import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchtext
import torchtext.vocab as vocab_module

from torchmetrics.classification import BinaryF1Score

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

from string import punctuation

from datasets import load_dataset
import csv

from functools import partial
import random

import math

class RTDataset(Dataset):
    """Rotten Tomatoes Dataset.

    This dataset class loads the Rotten Tomatoes dataset using the Hugging Face datasets library.
    It provides access to the text documents and their corresponding labels.

    Args:
        split (str): Specifies the split of the dataset ('train', 'validation', or 'test').
    """

    def __init__(self, split: str):
        """Initialize the RTDataset.

        Args:
            split (str): Specifies the split of the dataset ('train', 'validation', or 'test').
        """
        rt = load_dataset('rotten_tomatoes', split=split)

        self.data = rt['text']  # Documents
        self.labels = rt['label']  # Classes

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """Get a data sample and its corresponding label.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the index, the data sample and its corresponding label.
        """
        return index, self.data[index], self.labels[index]
    
class Preprocessor:
    """Preprocessor for text data.

    This preprocessor tokenizes, filters, and converts input text into indices
    for further processing, such as training a model.

    Args:
        max_length (int, optional): Maximum length of input sequences. Defaults to 50.
        embed_dim (int, optional): Dimensionality of word embeddings. Defaults to 100.
    """
    def __init__(self, max_length=50, embed_dim=100):
        """Initialize the Preprocessor.

        Args:
            max_length (int, optional): Maximum length of input sequences. Defaults to 50.
            embed_dim (int, optional): Dimensionality of word embeddings. Defaults to 100.
        """
        self.vocab = vocab_module.GloVe(name='6B', dim=embed_dim)  # Initialize GloVe vocabulary
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(punctuation)
        self.functional_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'from', 'of', 'for', 'with', 'by', 'about', 'as', 'into', 'onto', 'upon', 'before', 'after', 'between', 'during', 'through', 'above', 'below', 'under', 'behind', 'beneath', 'beside', 'among', 'within', 'without', 'though', 'although', 'because', 'since', 'until', 'while', 'if', 'then', 'else', 'either', 'neither', 'nor', 'not', 'so', 'and', 'but', 'or', 'as', 'if', 'whether', 'until', 'unless', 'than'}

        # Add special tokens to the vocabulary
        self.vocab.itos.append('<unk>')
        self.vocab.stoi['<unk>'] = len(self.vocab.itos) - 1

        self.vocab.itos.append('<pad>')
        self.vocab.stoi['<pad>'] = len(self.vocab.itos) - 1

        self.unk_vector = torch.zeros(1, self.vocab.dim) # <- Average of all 400k glove vectors versus zero vector.
        # Xavier Uniform Random if that.
        self.pad_vector = torch.zeros(1, self.vocab.dim)

        # Create a new embedding matrix with the updated vocabulary size
        updated_embeddings = torch.cat([self.vocab.vectors, self.unk_vector], dim=0)

        # Replace the original embedding matrix with the updated one
        self.vocab.vectors = updated_embeddings

        # Create a new embedding matrix with the updated vocabulary size
        updated_embeddings = torch.cat([self.vocab.vectors, self.pad_vector], dim=0)

        # Replace the original embedding matrix with the updated one
        self.vocab.vectors = updated_embeddings


    def preprocess_text(self, text):
        """Preprocess input text.

        This function tokenizes the input text, removes stopwords, punctuation,
        and functional words, and returns the filtered list of words.

        Args:
            text (str): Input text to be preprocessed.

        Returns:
            list: List of preprocessed words.
        """
        # Tokenize the text
        words = nltk.word_tokenize(text.lower())

        # Remove punctuations, stop words, and functional words
        filtered_words = [word for word in words if word not in self.punctuation and
                          word not in self.stop_words and word not in self.functional_words]

        return filtered_words

    def words_to_indices(self, words):
        """Convert words to corresponding indices.

        This function maps words to their corresponding indices in the vocabulary.
        Unknown words are assigned the index of the '<unk>' token.

        Args:
            words (list): List of words to be converted to indices.

        Returns:
            list: List of indices corresponding to the input words.
        """
        indices = [self.vocab.stoi.get(word, self.vocab.stoi['<unk>']) for word in words]
        return indices

    def pad_or_truncate(self, indices):
        """Pad or truncate indices to match max_length.

        This function ensures that the length of the input indices matches the
        specified max_length by padding or truncating the sequence.

        Args:
            indices (list): List of input indices.

        Returns:
            list: List of padded or truncated indices.
        """
        # Check if the length of indices exceeds max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        elif len(indices) < self.max_length:
            # Pad with <pad> token if length is less than max_length
            indices.extend([self.vocab.stoi['<pad>']] * (self.max_length - len(indices)))
        return indices

    def __call__(self, batch):
        """Perform preprocessing on a batch of data.

        This function applies preprocessing steps to a batch of data,
        including tokenization, conversion to indices, and padding/truncating.

        Args:
            batch (list): List of tuples (idx, X, Y) representing data samples.

        Returns:
            tuple: Tuple containing indices, input data (X), and labels (Y).
        """
        idx, X, Y = list(zip(*batch))

        X = [self.preprocess_text(x) for x in X]
        X = [self.words_to_indices(x) for x in X]
        X = [self.pad_or_truncate(x) for x in X]

        return idx, torch.tensor(X).float(), torch.tensor(Y).float()

class StatisticalPreprocessor:
    """Statistical Preprocessor for TF-IDF and Word Embedding Representations.

    This preprocessor computes TF-IDF and word embedding representations for input documents.

    Args:
        glove (vocab_module.vectors.GloVe): GloVe embeddings for word embedding representation.
    """
    def __init__(self, glove: vocab_module.vectors.GloVe):
        """Initialize the StatisticalPreprocessor.

        Args:
            glove (vocab_module.vectors.GloVe): GloVe embeddings for word embedding representation.
        """
        # Load GloVe embeddings
        self.glove = glove
        self.dim = glove.dim
        
    def tfidf_representation(self, documents):
        """Compute TF-IDF representation for input documents.

        Args:
            documents (List[List[int]]): List of documents, where each document is represented as a list of word indices.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF matrix representation of input documents.
        """
        # Convert word indices to strings for each document
        documents_str = [' '.join(map(str, doc)) for doc in documents]

        # Create CountVectorizer to compute term frequencies
        count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        count_matrix = count_vectorizer.fit_transform(documents_str)

        # Create TF-IDF transformer
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

        return tfidf_matrix

    def word_embedding_representation(self, documents):
        """Compute word embedding representation for input documents.

        Args:
            documents (List[List[int]]): List of documents, where each document is represented as a list of word indices.

        Returns:
            torch.Tensor: Word embedding representation of input documents.
        """
        # Initialize empty list to store embeddings
        embeddings = []

        # Iterate over each document
        for doc in documents:
            # Map GloVe indices to tokens

            tokens = [self.glove.itos[token_idx] for token_idx in doc]

            # Get embeddings for each token
            token_embeddings = [self.glove[token] for token in tokens]

            # Append embeddings to list
            embeddings.append(token_embeddings)

        # Stack each inner list of tensors individually
        embeddings = [torch.stack(inner_list, dim=0) for inner_list in embeddings]

        # Stack the list of stacked inner lists along a new dimension (dimension 1)
        embeddings = torch.stack(embeddings, dim=0)

        return embeddings

    def factor(self, vectors, scalars):
        """Factorize word embeddings with TF-IDF scalars.

        Args:
            vectors (torch.Tensor): Word embedding representation of input documents.
            scalars (scipy.sparse.csr_matrix): TF-IDF scalar representation of input documents.

        Returns:
            torch.Tensor: Factorized word embeddings.
        """
        # Convert sparse array to dense array
        dense_array = scalars.toarray()

        # Get the number of non-zero entries in the sparse array
        num_nonzero = len(scalars.data)

        # Get the maximum index of the second dimension of the vectors tensor
        max_index = vectors.shape[1]

        # Initialize a mapping from the original index to the sequential index
        index_mapping = {}

        # Initialize the sequential index
        sequential_index = 0

        # Iterate through non-zero entries in the sparse array
        for i, j, value in zip(scalars.nonzero()[0], scalars.nonzero()[1], scalars.data):
            # If the original index hasn't been encountered before, assign it a sequential index
            if j not in index_mapping:
                if sequential_index >= max_index:
                    break  # Stop if we reach the maximum index
                index_mapping[j] = sequential_index
                sequential_index += 1

            # Update the second dimension index to the sequential index
            j_new = index_mapping[j]

            # Multiply the corresponding elements in the tensor by the scalar value
            vectors[i, j_new] *= value

        return vectors

    def __call__(self, data):
        """Apply preprocessing steps to input data.

        Args:
            data (List[List[int]]): List of documents, where each document is represented as a list of word indices.

        Returns:
            torch.Tensor: Processed data.
        """
        # Get TF-IDF representation
        tfidf_matrix = self.tfidf_representation(data)

        # Get word embedding representation
        word_embedding_tensor = self.word_embedding_representation(data)

        # Apply factorization
        x = self.factor(word_embedding_tensor, tfidf_matrix)

        # Return processed data and labels
        return torch.tensor(x)

class SemanticPreprocessor:
    """Semantic Preprocessor for Word Embedding Representation.

    This preprocessor computes word embedding representations for input documents.

    Args:
        glove (torchtext.vocab.vectors.GloVe): GloVe embeddings for word embedding representation.
    """
    def __init__(self, glove: torchtext.vocab.vectors.GloVe):
        """Initialize the SemanticPreprocessor.

        Args:
            glove (torchtext.vocab.vectors.GloVe): GloVe embeddings for word embedding representation.
        """
        # Load GloVe embeddings
        self.glove = glove
        self.dim = self.glove.dim

    def word_embedding_representation(self, documents):
        """Compute word embedding representation for input documents.

        Args:
            documents (List[List[int]]): List of documents, where each document is represented as a list of word indices.

        Returns:
            torch.Tensor: Word embedding representation of input documents.
        """
        # Initialize empty list to store embeddings
        embeddings = []

        # Iterate over each document
        for doc in documents:
            # Map GloVe indices to tokens

            tokens = [self.glove.itos[token_idx] for token_idx in doc]

            # Get embeddings for each token
            token_embeddings = [self.glove[token] for token in tokens]

            # Append embeddings to list
            embeddings.append(token_embeddings)

        # Stack each inner list of tensors individually
        embeddings = [torch.stack(inner_list, dim=0) for inner_list in embeddings]

        # Stack the list of stacked inner lists along a new dimension (dimension 1)
        embeddings = torch.stack(embeddings, dim=0)

        return embeddings

    def __call__(self, data):
        """Apply preprocessing steps to input data.

        Args:
            data (torch.Tensor): Input data, where each element represents a document as a list of word indices.

        Returns:
            torch.Tensor: Processed data.
        """
        x = self.word_embedding_representation(data.int())

        return torch.tensor(x)

class FullyConnectedNet(pl.LightningModule):
    """
    A fully connected neural network with customizable architecture.

    Args:
        input_size (int): The size of the input features.
        output_size (int, optional): The size of the output. Defaults to 1.
        dropout (bool, optional): Whether to apply dropout. Defaults to True.
        dropout_ratio (float, optional): The dropout ratio. Defaults to 0.4.
        batch_norm (bool, optional): Whether to apply batch normalization. Defaults to True.
    """
    def __init__(self, input_size, output_size=1, dropout=True, dropout_ratio=0.4, batch_norm=True):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear(int(input_size/2), int(input_size/4))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.clh = nn.Linear(int(input_size/4), output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        self.batch_norm = batch_norm

        if self.dropout:
            self.dropout1 = nn.Dropout(self.dropout_ratio)
            self.dropout2 = nn.Dropout(self.dropout_ratio)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(int(input_size/2))
            self.bn2 = nn.BatchNorm1d(int(input_size/4))

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.relu1(self.fc1(x))
        if self.batch_norm:
            x = self.bn1(x)
        if self.dropout:
            x = self.dropout1(x)

        x = self.relu2(self.fc2(x))
        if self.batch_norm:
            x = self.bn2(x)
        if self.dropout:
            x = self.dropout2(x)

        x = self.sigmoid(self.clh(x))
        return x

class LocalSemanticNet(pl.LightningModule):
    """
    A convolutional neural network for local semantic analysis.

    Args:
        embed_dim (int, optional): The dimension of word embeddings. Defaults to 100.
        doc_len (int, optional): Length of the document. Defaults to 50.
        out_channels (int, optional): The number of output channels for each convolutional layer. Defaults to 96.
        kernel_sizes (tuple of int, optional): The sizes of convolutional kernels. Defaults to (1, 2, 3).
        out_dim (int, optional): Dimensionality of the output. Defaults to 100.
    """

    def __init__(self, embed_dim=100, doc_len=50, out_channels=96, kernel_sizes=(1, 2, 3), out_dim=100):
        """
        Initializes the LocalSemanticNet model.

        Args:
            embed_dim (int, optional): The dimension of word embeddings. Defaults to 100.
            doc_len (int, optional): Length of the document. Defaults to 50.
            out_channels (int, optional): The number of output channels for each convolutional layer. Defaults to 96.
            kernel_sizes (tuple of int, optional): The sizes of convolutional kernels. Defaults to (1, 2, 3).
            out_dim (int, optional): Dimensionality of the output. Defaults to 100.
        """
        super(LocalSemanticNet, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=out_channels, kernel_size=k) for k in kernel_sizes])
        self.maxpools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])
        self.conv_batch_norms = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in kernel_sizes])
        
        input_size = int(sum([(math.floor((doc_len - k + 1) / 2)) * out_channels for k in kernel_sizes]))

        self.fc = nn.Linear(input_size, out_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim, doc_len).

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        pooled_feature_maps = []
        x = x.permute(0, 2, 1)  # Transforms into (batch_size, embed_dim, doc_len)

        for conv, maxpool, bn in zip(self.convs, self.maxpools, self.conv_batch_norms):
            feature_map = conv(x)
            feature_map = bn(feature_map)  # Apply batch normalization
            pooled_feature_map = maxpool(F.relu(feature_map))  # Apply ReLU after batch normalization
            pooled_feature_maps.append(pooled_feature_map)

        flattened = [i.view(i.size(0), -1) for i in tuple(pooled_feature_maps)]
        x = torch.cat(flattened, dim=1)
        x = self.fc(x)
        x = self.relu(x)

        return x

class OverallSemanticNet(pl.LightningModule):
    """
    A recurrent neural network for overall semantic analysis.

    Args:
        embed_dim (int, optional): The dimension of word embeddings. Defaults to 100.
        doc_len (int, optional): Length of the document. Defaults to 50.
        out_size (int, optional): The output size of the GRU layer. Defaults to 128.
        bidirectionality (bool, optional): Whether to use bidirectional GRU. Defaults to True.
        out_dim (int, optional): Dimensionality of the output. Defaults to 100.
    """

    def __init__(self, embed_dim=100, doc_len=50, out_size=128, bidirectionality=True, out_dim=100):
        """
        Initializes the OverallSemanticNet model.

        Args:
            embed_dim (int, optional): The dimension of word embeddings. Defaults to 100.
            doc_len (int, optional): Length of the document. Defaults to 50.
            out_size (int, optional): The output size of the GRU layer. Defaults to 128.
            bidirectionality (bool, optional): Whether to use bidirectional GRU. Defaults to True.
            out_dim (int, optional): Dimensionality of the output. Defaults to 100.
        """
        super(OverallSemanticNet, self).__init__()

        self.gru = nn.GRU(embed_dim, out_size, bidirectional=bidirectionality)

        if bidirectionality:
            self.fc = nn.Linear(out_size * doc_len * 2, out_dim)
        else:
            self.fc = nn.Linear(out_size * doc_len, out_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x, _ = self.gru(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)

        return x

class SemanticNet(pl.LightningModule):
    """
    A neural network for semantic analysis using both local and overall information.

    Args:
        local_out_size (int): The output size of the local semantic network.
        overall_out_size (int): The output size of the overall semantic network.
        embed_dim (int): The dimension of word embeddings.
    """

    def __init__(self,  embed_dim=100, doc_len=50, local_out_size=96, local_kernels=(1,2,3), 
                        overall_out_size=128, overall_bidirectionality=True, out_dim=100):
        super(SemanticNet, self).__init__()

        self.local = LocalSemanticNet(embed_dim=embed_dim, 
                                      doc_len=doc_len, 
                                      out_channels=local_out_size, 
                                      kernel_sizes=local_kernels, 
                                      out_dim=out_dim)
        
        self.overall = OverallSemanticNet(embed_dim=embed_dim, 
                                          doc_len=doc_len, 
                                          out_size=overall_out_size, 
                                          bidirectionality=overall_bidirectionality, 
                                          out_dim=out_dim)

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Combined output tensor from local and overall semantic networks.
        """
        local_output = self.local(x)
        overall_output = self.overall(x)

        combined_output = local_output + overall_output  # Element-wise addition

        return combined_output

class FusionNet(pl.LightningModule):
    """
    FusionNet combines semantic and statistical information for classification.
    """
    def __init__(self, semantics, statistics, 
                 learning_rate=1e-3, embed_dim=100, doc_len=50, local_out_size=96, local_kernels=(1, 2, 3), 
                 overall_out_size=128, overall_bidirectionality=True, 
                 out_dim=100):
      """
        Initializes the FusionNet model.

        Args:
            semantics: Preprocessor for semantic information.
            statistics: Preprocessor for statistical information.
            learning_rate (float): Learning rate for optimizer. Default is 1e-3.
            embed_dim (int): Dimensionality of the embedding. Default is 100.
            doc_len (int): Length of the document. Default is 50.
            local_out_size (int): Size of output for local convolution layers. Default is 96.
            local_kernels (tuple): Sizes of local convolutional kernels. Default is (1, 2, 3).
            overall_out_size (int): Size of output for overall convolutional layer. Default is 128.
            overall_bidirectionality (bool): Whether to use bidirectional overall convolution. Default is True.
            out_dim (int): Dimensionality of the output. Default is 100.
      """
      super(FusionNet, self).__init__()
      self.validation_step_outputs = []
      self.test_step_output = []

      self.learning_rate = learning_rate
      self.f1_score_val = BinaryF1Score()
      self.f1_score_test = BinaryF1Score()
      self.loss_fn = torch.nn.BCELoss()

      self.semantics = semantics
      self.statistics = statistics

      self.semantic_net = SemanticNet(embed_dim=embed_dim, doc_len=doc_len, local_out_size=local_out_size, local_kernels=local_kernels, 
                                      overall_out_size=overall_out_size, overall_bidirectionality=overall_bidirectionality, 
                                      out_dim=out_dim)
      self.fc_net = FullyConnectedNet(embed_dim * doc_len + out_dim)

      self.tp = torch.tensor([0])
      self.fp = torch.tensor([0])
      self.fn = torch.tensor([0])

    def semantics_branch(self, x):
      """
        Process the input tensor through the semantics branch.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the semantics branch.
      """
      x = self.semantics(x.int())
      return self.semantic_net(x)

    def statistics_branch(self, x):
        """
        Process the input tensor through the statistics branch.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the statistics branch.
        """
        return self.statistics(x.int())

    def forward(self, x):
        """
        Perform forward pass through the FusionNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        semantics_output = self.semantics_branch(x)
        statistics_output = self.statistics_branch(x)
        statistics_output = statistics_output.view(statistics_output.size(0), -1)

        x = torch.cat((semantics_output, statistics_output), dim=1)
        x = x.view(x.size(0), -1)

        x = self.fc_net(x)
        return x

    def training_step(self, batch):
        """
        Perform a training step.

        Args:
            batch (tuple): Batch of data containing input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        self.train()
        idx, x, y = batch
        y_hat = self(x)  # 0dim tensor [[1]] transformed by sigmoid.
        loss = self.loss_fn(y_hat, y.unsqueeze(1).float())  # Ensure y is the correct shape and type
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def calculate_f1_score_from_counts(self, tp, fp, fn):
      """
        Calculate F1 score from true positive, false positive, and false negative counts.

        Args:
            tp (int): True positive count.
            fp (int): False positive count.
            fn (int): False negative count.

        Returns:
            float: Calculated F1 score.
      """
      if tp == 0:
        return 0
    
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
    
      f1_score = 2 * (precision * recall) / (precision + recall)
      return f1_score

    def validation_step(self, batch):
        """
        Perform a validation step.

        Args:
            batch (tuple): Batch of data containing input and target tensors.
        Returns:
            torch.Tensor: Validation loss tensor.
        """
        self.eval()
        idx, x, y = batch
        y_hat = self(x) # 0dim tensor [[1]] transformed by sigmoid.
        val_loss = self.loss_fn(y_hat, y.unsqueeze(1).float()) # LogitLossFn
        self.validation_step_outputs.append(val_loss)

        y_pred = (y_hat >= 0.5).float()

        y = y.unsqueeze(1).float()
        self.f1_score_val.update(y_pred.cpu(), y.cpu())

        self.tp += self.f1_score_val.tp
        self.fn += self.f1_score_val.fn
        self.fn += self.f1_score_val.fn

        self.f1_score_val.reset()

        return val_loss

    def on_validation_epoch_end(self):
        """
        Run at the end of each validation epoch.
        """
        mean_bce_loss = torch.stack(self.validation_step_outputs).mean() # compute
        self.log('val_loss', mean_bce_loss, on_epoch=True, prog_bar=True, logger=True) #post
        self.validation_step_outputs = [] # reset.

        f1 = self.calculate_f1_score_from_counts(tp=self.tp, fp=self.fp, fn=self.fn)

        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)

        self.tp = torch.tensor([0])
        self.fp = torch.tensor([0])
        self.fn = torch.tensor([0])

    def test_step(self, batch):
        """
        Perform a test step.

        Args:
            batch (tuple): Batch of data containing input and target tensors.
        """
        self.eval()
        idx, x, y = batch
        y_hat = self(x) # 0dim tensor [[1]] transformed by sigmoid.
        y_pred = (y_hat >= 0.5).float()

        y = y.unsqueeze(1).float()
        self.f1_score_val.update(y_pred.cpu(), y.cpu())

        self.tp += self.f1_score_val.tp
        self.fn += self.f1_score_val.fn
        self.fn += self.f1_score_val.fn

        self.f1_score_val.reset()

        # Append batch index and predictions to test_step_output
        self.test_step_output.append((idx, y_pred.cpu().numpy()))

    def on_test_epoch_end(self):
        """
        Run at the end of each test epoch to calculate F1 score and save predictions to a CSV file.
        """
        report = f"True Positives: {self.tp}\n"
        report += f"False Positives: {self.fp}\n"
        report += f"False Negatives: {self.fn}\n"

        f1 = self.calculate_f1_score_from_counts(tp=self.tp, fp=self.fp, fn=self.fn)
        report += f"Test Macro F1 Score: {f1}\n"

        print(report)

        self.tp = torch.tensor([0])
        self.fp = torch.tensor([0])
        self.fn = torch.tensor([0])

        output_filename = "results.csv"
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for batch_idx, batch_pred in self.test_step_output:
                for idx, pred in zip(batch_idx, batch_pred):
                    writer.writerow([idx, int(pred)])  # Assuming pred is 0 or 1

        # Read predictions from the CSV file
        with open('results.csv', mode='r', newline='') as file:
            reader = csv.reader(file)
            predictions = list(reader)

        # Convert string indices to integers and predictions to integers
        predictions = [(int(row[0]), int(row[1])) for row in predictions]

        # Sort predictions by index in ascending order
        sorted_predictions = sorted(predictions, key=lambda x: x[0])

        with open('results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["index", "pred"])
            writer.writerows(sorted_predictions)
        
        print(f"results saved to {output_filename}")



    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def collate_and_transfer_to_device(batch, preprocessor, device):
    """
    Apply a preprocessor to a batch of data and transfer the processed tensors to the specified device.

    Args:
        batch (tuple): A batch of data where the first element is the input data and the second element is the target labels.
        preprocessor (callable): A function or callable object that preprocesses the batch of data.
        device (torch.device): The device to which the processed tensors will be transferred.

    Returns:
        tuple: A tuple containing the processed input data and target labels, both transferred to the specified device.
    """
    # Apply the preprocessor
    processed_batch = preprocessor(batch)

    # `processed_batch` is expected to be a tuple (idx, X, Y)
    # Move both tensors in the tuple to the specified device
    idx, X, Y = processed_batch  # Unpack the tuple
    X = X.to(device)
    Y = Y.to(device)

    # Return a tuple of the tensors, now both on the correct device
    return idx, X, Y

######MAIN_FUNCTION######

def main():
    seed_value = 42
    max_length = 30
    embed_dim = 100
    local_out_size = 96
    local_kernels = (1, 3, 4)
    overall_out_size = 128
    bidirectionality = True
    semantic_shape = 100
    batch_size=32
    device = 'cpu'

    # Set seed
    random.seed(seed_value)

    # Set seed for CPU
    torch.manual_seed(seed_value)

    # Set seed for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    nltk.download('stopwords')
    nltk.download('punkt')


    train = RTDataset('train')
    val = RTDataset('validation')
    test = RTDataset('test')

    preprocessor = Preprocessor(max_length=max_length, embed_dim=embed_dim)

    semantics = SemanticPreprocessor(preprocessor.vocab)
    statistics = StatisticalPreprocessor(preprocessor.vocab)

    model = FusionNet(semantics, statistics).to(device)

    model = FusionNet(semantics, statistics, 
                      learning_rate = 1e-3, embed_dim=embed_dim, doc_len=max_length, local_out_size=local_out_size, local_kernels=local_kernels,
                      overall_out_size=overall_out_size, overall_bidirectionality=bidirectionality, out_dim=semantic_shape)

    for param in model.parameters():
        param.requires_grad_(True)

    collate_fn_device = partial(collate_and_transfer_to_device,
                            preprocessor=preprocessor,
                            device=device)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_device)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_device)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_device)

    # Initialize the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_f1',    # or 'val_f1' to monitor F1 score
        patience=5,           # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='max'             # 'min' because we want to minimize validation loss; use 'max' for metrics that should be maximized
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_model',
        monitor='val_f1',  # Metric to monitor for saving the best model
        mode='max',  # 'min' if the monitored metric should be minimized, 'max' if it should be maximized
        save_top_k=1,  # Save the top 1 best models
    )

    trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stopping])
    model.train()
    trainer.fit(model, train_loader, val_loader)

    loaded_model = torch.load('checkpoints/best_model.ckpt', map_location=torch.device(device))

    model.load_state_dict(loaded_model['state_dict'])
    model.eval()
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
