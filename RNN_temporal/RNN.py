import torch

# Model class
class RNN(torch.nn.Module):

    def __init__(self):
        super(RNN, self).__init__()  

        self.LSTM = torch.nn.LSTM(input_size = 2048, hidden_size = 1024, batch_first=True) # 1xNx2048 -> 1xNx1024
        self.fc_1 = torch.nn.Linear(1024, 512) # 1xNx1024 -> 1xNx512
        self.relu_1 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(512, 11) # 1xNx512 -> 1xNx11  

    def forward(self, features):
        
        # seq_len = timestep
        # features = (seq_len, batch, num_features)
        # x = (batch, sequence, hidden_size)

        x, (_, _) = self.LSTM(features)
        #print("After LSTM:")
        #print(x.shape)

        x = self.fc_1( x[:, :, :] )  
        #print("After first linear:")
        #print(x.shape)

        x = self.relu_1(x)
        #print("After ReLU:")

        x = self.fc_2(x)
        #print("After second linear:")
        #print(x.shape)

        return x
    
    def return_features(self, features):
        x, (_, _) = self.LSTM(features)
        x = x[:, -1, :]
        return x