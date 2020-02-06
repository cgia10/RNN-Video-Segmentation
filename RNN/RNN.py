import torch

# Model class
class RNN(torch.nn.Module):

    def __init__(self):
        super(RNN, self).__init__()  

        self.LSTM = torch.nn.LSTM(input_size = 2048, hidden_size = 1024, batch_first=True) # 1xNx2048 -> 1xNx1024
        self.fc_1 = torch.nn.Linear(1024, 512) # 1xNx1024 -> 1x512 (taking last sequence number)
        self.relu_1 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(512, 11) # 1x512 -> 1x11  

    def forward(self, features):
        
        # seq_len = timestep
        # features = (seq_len, batch, num_features)
        # x = (batch, sequence, hidden_size)

        x, (_, _) = self.LSTM(features)
        #print("After LSTM:")
        #print(x.shape)

        #self.fc_1.flatten_parameters()
        x = self.fc_1( x[:, -1, :] )  
        #print("After first linear:")
        #print(x.shape)

        #self.relu_1.flatten_parameters()
        x = self.relu_1(x)
        #print("After ReLU:")

        #self.fc_2.flatten_parameters()
        x = self.fc_2(x)
        #print("After second linear:")
        #print(x.shape)

        return x
    
    def return_features(self, features):
        x, (_, _) = self.LSTM(features)
        x = x[:, -1, :]
        return x