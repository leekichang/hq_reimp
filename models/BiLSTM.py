import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Stacked_BiLSTM(nn.Module):
    def __init__(self, batch_size, input_size, num_layers, hidden_size, num_direction):
        super().__init__()
        self.input_size  = input_size
        self.batch_size  = batch_size
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.num_direction = num_direction
        
        self.bilstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=self.hidden_size*self.num_direction, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=True, batch_first=True)
        
        self.fc1 = nn.Linear(self.hidden_size*self.num_direction*(750//self.input_size), self.hidden_size*2)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, 750)
        
    def init_hidden(self, num_layers, num_direction, batch_size, hidden_size):
        hidden_state = torch.zeros(num_layers*num_direction, batch_size, hidden_size).to(DEVICE)
        cell_state   = torch.zeros(num_layers*num_direction, batch_size, hidden_size).to(DEVICE)
        return hidden_state, cell_state
        
    def forward(self, x):
        hs_1, cs_1 = self.init_hidden(self.num_layers, self.num_direction, self.batch_size, self.hidden_size)
        x         = x.view(self.batch_size, -1, self.input_size)
        x_bilstm1, (output_hs, output_cs) = self.bilstm1(x, (hs_1, cs_1))
        x_bilstm2, (output_hs, output_cs) = self.bilstm2(x_bilstm1, (output_hs, output_cs))
        x_fc1 = self.fc1(x_bilstm2.contiguous().view(self.batch_size, -1))
        x_fc2 = self.fc2(x_fc1)
        x_fc3 = self.fc3(x_fc2)
        
        return x_fc3
    
if __name__ == '__main__':
    BATCH_SIZE     = 256
    INPUT_SIZE     = 50
    NUM_LAYERS     = 1
    LSTM_HS        = 512
    NUM_DIRECTION  = 2
    MODEL = Stacked_BiLSTM(BATCH_SIZE, INPUT_SIZE, NUM_LAYERS, LSTM_HS, NUM_DIRECTION)
    MODEL = MODEL.to(DEVICE)

    x = torch.randn(BATCH_SIZE, 750, INPUT_SIZE).to(DEVICE)
    y = MODEL(x)
    print(x.shape, y.shape)