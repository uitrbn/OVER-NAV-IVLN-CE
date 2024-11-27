import gzip
import json
import numpy as np

import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor


class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        if self.config.sensor_uuid == "instruction": # true
            instruction = observations["instruction"].long()
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction) # (batch size, 200, 50)
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu() # token number for each batch sample

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)
        # final_state: tuple of size 2, final_state[0]: [2, batch_size x hidden_size], final_state[1]: [2, batch_size x hidden_size]

        if self.config.rnn_type == "LSTM": # true
            final_state = final_state[0] # [2, batch_size x hidden_size]

        if self.config.final_state_only: # false
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1) # [batch_size, hidden_size, max seq_length] [4, 256, 44]

    def keywords_forward(self, observations: Observations) -> Tensor:
        if 'keywords_map' in observations:
            batch_keywords_tokens_size = [x.shape[0] for x in observations["keywords_map"]["keywords"]]
            # print('Average Keywords Number: {}'.format(sum(batch_keywords_tokens_size) / len(batch_keywords_tokens_size)))
            batch_keywords_tokens = np.concatenate(observations["keywords_map"]["keywords"], axis=0)
        elif 'keywords' in observations:
            batch_keywords_tokens_size = [x.shape[0] for x in observations["keywords"]]
            # print('Average Keywords Number: {}'.format(sum(batch_keywords_tokens_size) / len(batch_keywords_tokens_size)))
            batch_keywords_tokens = np.concatenate(observations["keywords"], axis=0)
        else:
            return None
        batch_keywords_tokens = torch.from_numpy(batch_keywords_tokens).cuda().long()
        
        lengths = (batch_keywords_tokens != 0.0).long().sum(dim=1)
        batch_keywords_tokens = self.embedding_layer(batch_keywords_tokens)

        lengths = (batch_keywords_tokens != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu() # token number for each batch sample

        if batch_keywords_tokens.shape[0] == 0:
            return None
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            batch_keywords_tokens, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM": # true
            final_state = final_state[0] # [2, batch_size x hidden_size]

        # if self.config.final_state_only: # false
        if True:
            final_state = final_state.permute(1, 0, 2)
            final_state = final_state.reshape(final_state.shape[0], -1)
            final_state = torch.split(final_state, batch_keywords_tokens_size, dim=0)
            # final_state = nn.utils.rnn.pad_sequence(final_state, batch_first=True)
            return final_state
        else:
            res = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            res = res[:, -1, :]
            res = torch.split(res, batch_keywords_tokens_size, dim=0)
            return res