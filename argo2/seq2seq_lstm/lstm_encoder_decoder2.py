import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import dropout_, optim
import torch.nn.functional as F

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.decoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, input_size)

        self.drop_layer = nn.Dropout(p=dropout)

    def train_model(
        self, input_tensor, target_tensor, n_epochs, target_len, batch_size, 
        training_prediction = 'recursive', 
        teacher_forcing_ratio = 0.5, 
        learning_rate = 0.01, 
        dynamic_tf = False
    ):
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        # initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[0] / batch_size)
        print('test')

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.

                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[b: b + batch_size, :, :]
                    target_batch = target_tensor[b: b + batch_size, :, :]

                    # outputs tensor
                    outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[:, -1, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                                decoder_output = self.linear(decoder_output.squeeze(1))
                                outputs[:, t, :] = decoder_output
                                decoder_input = target_batch[:, t, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                                decoder_output = self.linear(decoder_output.squeeze(1))
                                outputs[:, t, :] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[:, t, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar 
                tr.set_postfix(loss="{0:.5f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(0)
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[:, -1, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
            decoder_output = self.linear(decoder_output.squeeze(0)) 

            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        out = outputs.detach().numpy()
        out = self.drop_layer(out)
        out = self.encoder(out)
        out = self.decoder(out)
        
        
        return out

class doubleLSTM(nn.Module):
    """
    The 1st LSTM predicts the initial position (first element) of the output trajectory.
    The 2nd LSTM predicts the velocities (delta x, y) (all elements) of the output trajectory.
    The output trajectory is thus obtained by stacking the output of the 1st LSTM and the output of the second LSTM and calling cumulative sum.
    """

    def __init__(self, input_size, hidden_size_init_pos, hidden_size_delta, num_layers=1, dropout=0):
        super(doubleLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size_init_pos = hidden_size_init_pos
        self.hidden_size_delta = hidden_size_delta
        self.num_layers = num_layers
        
        self.lstm_init_pos = lstm_seq2seq(input_size, hidden_size_init_pos, batch_first = True, dropout = dropout)
        self.lstm_delta = lstm_seq2seq(input_size, hidden_size_delta, batch_size = True, dropout = dropout)

    def predict(self, input_tensor, target_len):
        """
        docstring
        """
        init_pos = self.lstm_init_pos.predict(input_tensor, 1)
        all_deltas = self.lstm_delta.predict(input_tensor, target_len)

        return torch.cumsum(torch.cat((init_pos, all_deltas)), dim=0)

    def train_model(
        self, input_tensor, target_tensor, n_epochs, target_len, batch_size, 
        training_prediction = 'recursive', 
        teacher_forcing_ratio = 0.5, 
        learning_rate = 0.01, 
        dynamic_tf = False
    ):
        losses = np.full(n_epochs, np.nan)
        optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()
        n_batches = int(input_tensor.shape[0] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                batch_loss = 0.
                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[b: b + batch_size, :, :]
                    target_batch = target_tensor[b: b + batch_size, :, :]

                    # outputs tensor
                    outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])

                    # zero the gradient
                    optimizer.zero_grad()





                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[:, -1, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                                decoder_output = self.linear(decoder_output.squeeze(1))
                                outputs[:, t, :] = decoder_output
                                decoder_input = target_batch[:, t, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                                decoder_output = self.linear(decoder_output.squeeze(1))
                                outputs[:, t, :] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[:, t, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar 
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses
