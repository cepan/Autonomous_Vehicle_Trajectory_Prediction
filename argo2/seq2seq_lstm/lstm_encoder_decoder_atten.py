from json import encoder
import numpy as np
import random
import time
from tqdm import trange

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
        # print('x: ', x.shape)
        # print('h: ', h.shape)

        # nn1
        # attn_weights = F.softmax(self.attn(torch.cat([x, h], 1)), dim = 1)
        # attn_applied = torch.einsum("ij,ijk->ik", attn_weights, encoder_outputs) 

        # nn2
        # dim(attn_weights) == (batch_size, input_len)
        # dim(encoder_outputs) == (batch_size, input_len, hidden_size)
        # dim(attn_applied) == (batch_size, hidden_size)
        # print(torch.cat([x,h], 1).shape)
        # print(self.attn(torch.cat([x,h], 1)).shape)
        attn_weights = F.softmax(self.attn(torch.cat([x, h], 1)), dim = 1)
        # print('dim(attn_weights): ', attn_weights.shape)
        # print('dim(encoder_outputs): ', encoder_outputs.shape)
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
    
    def __init__(self, input_size, input_len, hidden_size, output_size, num_layers=1, dropout=0):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, batch_first = True, dropout = dropout
        )
        # self.decoder = nn.LSTM(
        #     input_size = input_size, 
        #     hidden_size = hidden_size, 
        #     num_layers = num_layers, batch_first = True, dropout = dropout
        # )
        self.attn_decoder = AttentionDecoder(
            input_len = input_len, 
            output_size = output_size, 
            hidden_size = hidden_size,
            num_layers = num_layers, 
            dropout = dropout
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def train_model(
        self, input_tensor, target_tensor, val_input_tensor, val_target_tensor, n_epochs, target_len, batch_size, 
        patience = 5, 
        threshold = 1e-3,
        training_prediction = 'recursive', 
        teacher_forcing_ratio = 0.5, 
        learning_rate = 0.01, 
        dynamic_tf = False,
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
        attention = True
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, threshold=threshold, min_lr=1e-10, verbose=True)

        n_batches = int(input_tensor.shape[0] / batch_size)

        for epoch in range(n_epochs):
            start = time.time()
            batch_loss = 0.

            for b in range(n_batches):
                input_batch = input_tensor[b: b + batch_size, :, :]
                target_batch = target_tensor[b: b + batch_size, :, :]
                outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])
                
                optimizer.zero_grad()

                encoder_output, concat_hidden = self.encoder(input_batch)
                decoder_hidden = concat_hidden
                decoder_input = input_batch[:, -1, :]   # shape: (batch_size, input_size)

                if training_prediction == 'recursive':
                    if attention:
                        for t in range(target_len): 
                            decoder_hidden = (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))
                            decoder_output, decoder_hidden = self.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output
                    else:
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            decoder_input = decoder_output

                if training_prediction == 'mixed_teacher_forcing':                    
                    if attention:
                        for t in range(target_len):
                            decoder_hidden = (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))
                            # print(decoder_input.squeeze().shape)
                            # print(decoder_hidden[0].shape)
                            decoder_output, decoder_hidden = self.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                            outputs[:, t, :] = decoder_output
                            
                            if random.random() < teacher_forcing_ratio: 
                                decoder_input = target_batch[:, t, :]
                            else: 
                                decoder_input = decoder_output
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                            decoder_output = self.linear(decoder_output.squeeze(1))
                            outputs[:, t, :] = decoder_output
                            
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[:, t, :]
                            
                            else:
                                decoder_input = decoder_output

                # compute the loss 
                loss = criterion(outputs, target_batch)
                batch_loss += loss.item()
                
                # backpropagation
                loss.backward()
                optimizer.step()

            # get validation loss
            if attention:
                val_outputs = torch.zeros(val_input_tensor.shape[0], target_len, val_input_tensor.shape[2])
                encoder_output, concat_hidden = self.encoder(val_input_tensor)
                decoder_input = val_input_tensor[:, -1, :]
                decoder_hidden = concat_hidden
                for t in range(target_len):
                    decoder_hidden = (decoder_hidden[0].squeeze(), decoder_hidden[1].squeeze())
                    decoder_output, decoder_hidden = self.attn_decoder(decoder_input.squeeze(), decoder_hidden, encoder_output)
                    val_outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output
            else:
                val_outputs = torch.zeros(val_input_tensor.shape[0], target_len, val_input_tensor.shape[2])
                _, concat_hidden = self.encoder(val_input_tensor)
                decoder_input = val_input_tensor[:, -1, :]
                decoder_hidden = concat_hidden
                for t in range(target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                    decoder_output = self.linear(decoder_output.squeeze(1))
                    val_outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output

            # scheduler checks validation loss every epoch
            val_loss = criterion(val_outputs, val_target_tensor)
            scheduler.step(val_loss)

            batch_loss /= n_batches
            losses[epoch] = batch_loss

            # dynamic teacher forcing
            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

            end = time.time()
            if epoch % int(n_epochs / 10) == 0:
                print('Epoch ' + str(epoch) + ' train loss, validation loss, time: ' 
                + "{0:.5f}".format(batch_loss) + ', ' 
                + "{0:.5f}".format(val_loss.item()) + ', '
                + "{0:.5f}".format(end - start))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(0)
        _, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[:, -1, :]   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
            decoder_output = self.linear(decoder_output.squeeze(0)) 

            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        out = outputs.detach()#.numpy()        
        return out

