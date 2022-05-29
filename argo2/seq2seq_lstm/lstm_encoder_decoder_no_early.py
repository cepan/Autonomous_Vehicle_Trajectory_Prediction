import numpy as np
import random
import os
import errno
import sys
from tqdm import trange
import copy
import gc

import pickle
import torch
import torch.nn as nn
from torch import dropout_, optim
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    def __init__(self, input_len, output_size, hidden_size, num_layers=1, dropout=0):
        super(AttentionDecoder, self).__init__() 
        # learn attention scores
        self.attn = nn.Linear(hidden_size + output_size, input_len)
        # learn attention scores to decoder
        self.attn_combine = nn.Linear(hidden_size + output_size, hidden_size)
        # decoder LSTM
        self.lstm = nn.LSTM(
            input_size = hidden_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True, 
            dropout = dropout
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs):
        h = hidden[0]
        h = h.reshape(h.shape[1], -1).transpose(1,0)

        attn_weights = F.softmax(self.attn(torch.cat([x, h], 1)), dim = 1)

        attn_applied = torch.einsum("ij,ijk->ik", attn_weights, encoder_outputs)      
        
        x = torch.cat((x, attn_applied), dim = 1)
        x = self.attn_combine(x).unsqueeze(1)
        x = F.relu(x)
        
        hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        output, decoder_hidden = self.lstm(x, hidden)  
        prediction = self.output_layer(output.float())
        
        return prediction.squeeze(1), decoder_hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, input_len, output_size, hidden_size, num_layers=1, dropout=0):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)

        # self.decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                        num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.attn_decoder = AttentionDecoder(
            input_len = input_len, 
            output_size = output_size, 
            hidden_size = hidden_size,
            num_layers = num_layers, 
            dropout = dropout
        )

        self.linear = nn.Linear(hidden_size, input_size)

        self.drop_layer = nn.Dropout(p=dropout)


def predict(model, input_tensor, target_len, device='cpu'):
    '''
    : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
    : param target_len:        number of target values to predict 
    : return np_outputs:       np.array containing predicted values; prediction done recursively 
    '''
    model.to(device)
    model.eval()

    # encode input_tensor
    input_tensor = input_tensor.unsqueeze(0)
    encoder_output, encoder_hidden = model.encoder(input_tensor)

    # initialize tensor for predictions
    outputs = torch.zeros(target_len, input_tensor.shape[2]).to(device)

    # decode input_tensor
    decoder_input = input_tensor[:, -1, :]
    decoder_hidden = encoder_hidden

    for t in range(target_len):
        decoder_output, decoder_hidden = model.decoder(
            decoder_input.unsqueeze(0), decoder_hidden)
        decoder_output = model.linear(decoder_output.squeeze(0))

        outputs[t] = decoder_output.squeeze(0)
        decoder_input = decoder_output

    out = outputs.detach()
    # out = model.encoder(out)
    # out = model.decoder(out)
    
    return out


def train_model(
        model, city, input_tensor_train, target_tensor_train, input_tensor_val, target_tensor_val,
        n_epochs, target_len, batch_size,
        patience = 5, 
        threshold = 1e-3,
        training_prediction='recursive',
        teacher_forcing_ratio=0.5,
        learning_rate=0.01,
        dynamic_tf=False,
        early_stop_criteria=20,
        device='cpu',
        attention = True,
        model_type = 'pos'):
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
    val_losses = []
    best_valid_loss = float('inf')
    early_stop_counter = 0
    
    # initialize optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, threshold=threshold, min_lr=1e-10, verbose=True)

    # calculate number of batch iterations
    n_batches_train = int(input_tensor_train.shape[0] / batch_size)
    n_batches_val = int(input_tensor_val.shape[0] / batch_size)


    with trange(n_epochs) as tr:

        for it in tr:

            model.train()
            batch_loss_train = 0.
            batch_loss_val = 0.

            for b in range(n_batches_train):
                # select data
                input_batch = input_tensor_train[b: b + batch_size, :, :]
                target_batch = target_tensor_train[b: b + batch_size, :, :]

                # outputs tensor
                outputs = torch.zeros(
                    batch_size, target_len, input_batch.shape[2]).to(device)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, concat_hidden = model.encoder(input_batch)

                # decoder with teacher forcing
                # shape: (batch_size, input_size)
                decoder_input = input_batch[:, -1, :]
                decoder_hidden = concat_hidden

                if training_prediction == 'recursive':
                    # predict recursively
                    if attention:
                        for t in range(target_len): 
                            decoder_hidden = (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))
                            decoder_output, decoder_hidden = model.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = model.decoder(
                                decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = model.linear(
                                decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output

                # if training_prediction == 'teacher_forcing':
                #     # use teacher forcing
                #     if random.random() < teacher_forcing_ratio:
                #         for t in range(target_len):
                #             decoder_output, decoder_hidden = model.decoder(
                #                 decoder_input.unsqueeze(1), decoder_hidden)
                #             decoder_output = model.linear(
                #                 decoder_output.squeeze(1))
                #             outputs[:, t, :] = decoder_output
                #             decoder_input = target_batch[:, t, :]

                #     # predict recursively
                #     else:
                #         for t in range(target_len):
                #             decoder_output, decoder_hidden = model.decoder(
                #                 decoder_input.unsqueeze(1), decoder_hidden)
                #             decoder_output = model.linear(
                #                 decoder_output.squeeze(1))
                #             outputs[:, t, :] = decoder_output
                #             decoder_input = decoder_output

                if training_prediction == 'mixed_teacher_forcing':

                    # predict using mixed teacher forcing

                    if attention:
                        for t in range(target_len):
                            decoder_hidden = (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))
                            # print(decoder_input.squeeze().shape)
                            # print(decoder_hidden[0].shape)
                            decoder_output, decoder_hidden = model.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                            outputs[:, t, :] = decoder_output
                            
                            if random.random() < teacher_forcing_ratio: 
                                decoder_input = target_batch[:, t, :]
                            else: 
                                decoder_input = decoder_output
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = model.decoder(
                                decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = model.linear(
                                decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[:, t, :]

                            # predict recursively
                            else:
                                decoder_input = decoder_output

                # compute the loss
                target_batch = target_batch.to(device)
                loss_train = criterion(outputs, target_batch)
                batch_loss_train += loss_train.item()

                # backpropagation
                loss_train.backward()
                optimizer.step()

            # loss for epoch
            batch_loss_train /= n_batches_train
            train_losses.append(batch_loss_train)


            # scheduler checks validation loss every epoch
            
            model.eval()
            for b in range(n_batches_val):
                # select data
                input_batch = input_tensor_val[b: b + batch_size, :, :]
                target_batch = target_tensor_val[b: b + batch_size, :, :]

                # outputs tensor
                outputs = torch.zeros(
                    batch_size, target_len, input_batch.shape[2]).to(device)

                # encoder outputs
                encoder_output, concat_hidden = model.encoder(input_batch)

                # decoder with teacher forcing
                # shape: (batch_size, input_size)
                decoder_input = input_batch[:, -1, :]
                decoder_hidden = concat_hidden

 
                if training_prediction == 'mixed_teacher_forcing':

                    # predict using mixed teacher forcing

                    if attention:
                        for t in range(target_len):
                            decoder_hidden = (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))
                            # print(decoder_input.squeeze().shape)
                            # print(decoder_hidden[0].shape)
                            decoder_output, decoder_hidden = model.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                            outputs[:, t, :] = decoder_output
                            
                            if random.random() < teacher_forcing_ratio: 
                                decoder_input = target_batch[:, t, :]
                            else: 
                                decoder_input = decoder_output

                # compute the loss
                target_batch = target_batch.to(device)
                loss_val = criterion(outputs, target_batch)
                batch_loss_val += loss_val.item()

            # loss for epoch
            batch_loss_val /= n_batches_val
            val_losses.append(batch_loss_val)
            scheduler.step(batch_loss_val)


            if batch_loss_val < best_valid_loss:
                best_valid_loss = batch_loss_val
                best_model = copy.deepcopy(model)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter > early_stop_criteria:
                    print("Stopped early at epoch " +
                          str(it) + " due to overfit")
                    pickle.dump(best_model, open(
                        '../models/seq2seq_lstm_' + model_type +'_atten_' + str(model.num_layers) + '_' + city, 'wb'))
                    break


            # dynamic teacher forcing
            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = teacher_forcing_ratio - 0.02



            # progress bar
            tr.set_postfix(loss="{0:.5f}".format(batch_loss_train))



    return train_losses, val_losses
