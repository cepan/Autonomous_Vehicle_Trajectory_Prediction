import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import copy

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



def predict(model, input_tensor, target_len):

    '''
    : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
    : param target_len:        number of target values to predict 
    : return np_outputs:       np.array containing predicted values; prediction done recursively 
    '''
    model.eval()

    # encode input_tensor
    input_tensor = input_tensor.unsqueeze(0)
    encoder_output, encoder_hidden = model.encoder(input_tensor)

    # initialize tensor for predictions
    outputs = torch.zeros(target_len, input_tensor.shape[2])

    # decode input_tensor
    decoder_input = input_tensor[:, -1, :]
    decoder_hidden = encoder_hidden

    for t in range(target_len):
        decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(0), decoder_hidden)
        decoder_output = model.linear(decoder_output.squeeze(0)) 

        outputs[t] = decoder_output.squeeze(0)
        decoder_input = decoder_output

    out = outputs.detach().numpy()
    out = model.drop_layer(out)
    out = model.encoder(out)
    out = model.decoder(out)


    return out


def train_model(
    model, input_tensor_train, target_tensor_train, input_tensor_val, target_tensor_val,
    n_epochs, target_len, batch_size, 
    training_prediction = 'recursive', 
    teacher_forcing_ratio = 0.5, 
    learning_rate = 0.01, 
    dynamic_tf = False,
    early_stop_criteria = 20,
    device='cpu'):
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
    : param early_stop_criteria:       int = 0; stop when n validation loss are not decreasing
    : param device:                    train on cpu or gpu
    : return train_losses:                   array of loss function for each epoch
    '''

    # initialize array of train_losses 
    train_losses = []
    # initialize array of valid_losses 
    valid_losses = []
    best_valid_loss = float('inf')

    best_model = None
    optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()

    # calculate number of batch iterations
    n_batches_train = int(input_tensor_train.shape[0] / batch_size)
    n_batches_val = int(input_tensor_val.shape[0] / batch_size)

    with trange(n_epochs) as tr:
        model.train()

        for it in tr:

            batch_loss_train = 0.

            for b in range(n_batches_train):
                # select data 
                input_batch = input_tensor_train[b: b + batch_size, :, :]
                target_batch = target_tensor_train[b: b + batch_size, :, :]


                # outputs tensor
                outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = model.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[:, -1, :]   # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                if training_prediction == 'recursive':
                    # predict recursively
                    for t in range(target_len): 
                        decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                        decoder_output = model.linear(decoder_output.squeeze(1))
                        outputs[:, t, :] = decoder_output
                        decoder_input = decoder_output

                if training_prediction == 'teacher_forcing':
                    # use teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = model.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = target_batch[:, t, :]

                    # predict recursively 
                    else:
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = model.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output

                if training_prediction == 'mixed_teacher_forcing':

                    # predict using mixed teacher forcing
                    for t in range(target_len):
                        decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                        decoder_output = model.linear(decoder_output.squeeze(1))
                        outputs[:, t, :] = decoder_output

                        # predict with teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            decoder_input = target_batch[:, t, :]

                        # predict recursively 
                        else:
                            decoder_input = decoder_output

                # compute the loss 
                loss_train = criterion(outputs, target_batch)
                batch_loss_train += loss_train.item()

                # backpropagation
                loss.backward()
                optimizer.step()

            # loss for epoch 
            batch_loss_train /= n_batches_train
            train_losses.append(batch_loss_train)

            # dynamic teacher forcing
            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = teacher_forcing_ratio - 0.02 


            model.eval()
            batch_loss_val = 0.

            for b in range(n_batches_val):
                # select data 
                input_batch = input_tensor_val[b: b + batch_size, :, :]
                target_batch = target_tensor_val[b: b + batch_size, :, :]

                input_batch = input_batch
                target_batch = target_batch

                # outputs tensor
                outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = model.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[:, -1, :]   # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden


                for t in range(target_len): 
                    decoder_output, decoder_hidden = model.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                    decoder_output = model.linear(decoder_output.squeeze(1))
                    outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output


                # compute the loss 
                loss_val = criterion(outputs, target_batch)
                batch_loss_val += loss_val.item()
                
            batch_loss_val /= n_batches_val
            valid_losses.append(batch_loss_val)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter > early_stop_criteria:
                    print("Stopped early at epoch " + str(it) + " due to overfit")
                    pickle.dump(best_model, open('./models/seq2seq_lstm_' + str(model.num_layers) + '_' + city, 'wb'))
                    break

            # progress bar 
            tr.set_postfix(loss="{0:.5f}".format(batch_loss))

    return train_losses, valid_losses
    