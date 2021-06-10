import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, image_resolution=64):
        super(Encoder, self).__init__()

        # assume image_resolution=64
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)] # input: 64*64*1
        )

        self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1))  # 32*32*4
        self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1))  # 16*16*8
        self.convs.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)) # 8*8*16
        self.convs.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)) # 4*4*32
        self.convs.append(nn.Conv2d(64, 128, kernel_size=2, stride=2)) # 2*2*64

        # the final 1x1 feature vector gets mapped to a 64-dimensional observation space
        self.fc = nn.Linear(in_features=128, out_features=64)  # input: 1*1*128 output: 64

    # x is the observation at one time step
    def forward(self, x, detach=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(6):
            x = torch.relu(self.convs[i](x))
        out = self.fc(x.squeeze())

        # freeze the encoder
        if detach:
            out.detach()
        return out


# Build a 3-layer feedforward neural network
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)     
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        hidden_layer = self.relu(self.fc1(x))
        # hidden_layer = self.relu(self.fc2(hidden_layer))
        output_layer = self.fc3(hidden_layer)
        return output_layer

# Build a 2-layer feedforward neural network
class MLP_2(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=32, is_actor=True):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.is_actor = is_actor
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        hidden_layer = self.relu(self.fc1(x))
        if self.is_actor:
            # [-1, 1]
            output_layer = self.tanh(self.fc2(hidden_layer))
        else:
            # is_critic: [0, inf)
            output_layer = self.relu(self.fc2(hidden_layer))
        return output_layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Architecture 1
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)])
        # self.convs.append(nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1))
        # self.convs.append(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1))

        # Architecture 2
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)])
        # self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1))
        # self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1))

        # self.fc = nn.Linear(in_features=16*8*8, out_features=64)

        # Architecture 3
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 32, kernel_size=3, stride=2)])  # 31
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 29
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 27

        # self.fc = nn.Linear(in_features=32*27*27, out_features=64)

    # x is the observation at one time step
    def forward(self, x, detach=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(3):
            x = torch.relu(self.convs[i](x))

        # freeze
        if detach:
            x.detach()
        return x.view(-1, 32*27*27)
        # return x.flatten()


class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=64):
        super(CNN_MLP, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 32, kernel_size=3, stride=2)])  # 31
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 29
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 27

        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(3):
            x = torch.relu(self.convs[i](x))

        hidden_layer = self.relu(self.fc1(x.view(-1, 32*27*27)))
        output_layer = self.fc2(hidden_layer)
        # output_layer = self.fc3(hidden_layer)
        return output_layer


# Build a single-layer LSTM        
class LSTM(nn.Module):
    def __init__(self, sequence_length, input_dim=64, hidden_dim=64, layer_dim=1, output_dim=64, device=None):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.sequence_length = sequence_length
        self.device = device

        # batch_first=True => input/output tensors of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): # x should be (64,)
        # hidden states provide the information
        h0 = x[None, None, :].to(self.device) # (1, 1, 64)
        # input cells, batch_dim=1
        input = torch.zeros(self.layer_dim, self.sequence_length, self.hidden_dim).to(self.device) # (1, T, 64) 
        # initialize cell states with zeros
        c0 = torch.zeros(self.layer_dim, input.size(0), self.hidden_dim).to(self.device)

        out, _ = self.lstm(input, (h0, c0))
        # return out.squeeze(0) # return (T, hidden)
        return self.fc(out.squeeze(0))


class Model(nn.Module):
    def __init__(self, time_steps, frames, tasks, image_resolution, device):
        super(Model, self).__init__()
        self.encoder = Encoder().to(device)
        self.mlp = MLP(input_size=image_resolution*frames, output_size=image_resolution).to(device)
        self.lstm = LSTM(sequence_length=time_steps, device=device).to(device)
        self.T = time_steps
        self.N = frames
        self.K = tasks
        self.reward_heads = [MLP(input_size=image_resolution, output_size=1).to(device)] * tasks
        self.image_resolution = image_resolution
        self.device = device
        self.loss = nn.MSELoss()
        self.decoder = Decoder().to(device)

    # return the prediction of T future rewards given N conditioning frames
    def forward(self, obs):
        # first obtain the representation z_t at each time step using encoder
        z = []
        for frame in range(self.N):
            # obs corresponds to the observation at N conditioning frames: (N, 64, 64)
            # so obs[frame] is the observation at the current frame, represented as a (64, 64) tensor
            # the input to encoder should be in (1, 1, 64, 64)
            z_frame = self.encoder(obs[frame][None, None, :]) #tensor(64,)
            z.append(z_frame)
        z = torch.stack(z, dim=0) # (N, 64)

        z_mlp = self.mlp(z.flatten()) # (64,)
        h = self.lstm(z_mlp) # (T, 64)
                
        # then feed h to each reward head to predict the reward of all time steps for every task
        reward_predicted_tasks = []
        for task in range(self.K):
            # reward_heads is a list of K MLP's
            reward_head = self.reward_heads[task]   
            reward_predicted = []
            for t in range(self.T):
                r_t = reward_head(h[t])   
                reward_predicted.append(r_t) 
            reward_predicted = torch.stack(reward_predicted, dim=0).squeeze() # (T,) 
            reward_predicted_tasks.append(reward_predicted)
        reward_predicted_tasks = torch.stack(reward_predicted_tasks, dim=0) # should be (K, T)
        return reward_predicted_tasks
                       
    def criterion(self, reward_predicted, reward_targets):
        reward_predicted = reward_predicted.squeeze()
        assert reward_predicted.shape == reward_targets.shape
        return self.loss(reward_predicted, reward_targets)

    def test_decode(self, traj_images):
        output = []
        for t in range(self.T):
            # the input to encoder should be in (1, 1, 64, 64)
            # print("\\\\", traj_images[t].max())
            z_t = self.encoder(traj_images[t][None, None, :], detach=True) #tensor(64,)
            decoded_img = self.decoder(z_t).squeeze()
            # print("////", decoded_img.max())
            output.append(decoded_img.detach().cpu().numpy())

        return np.array(output)

# add a detached decoder network and train the model on a single reward
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.tfc = nn.Linear(64, 128)
        # self.tconvs = nn.ModuleList(
        #     [nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1)]
        # )

        # self.tconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.tconvs = nn.ModuleList(
            [nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)] 
        )

        self.tconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2))
        self.tconvs.append(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2))
        self.tconvs.append(nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2))
        self.tconvs.append(nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2))
        self.tconvs.append(nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2))

    def forward(self, x):
        x = self.tfc(x).view(-1, 128, 1, 1)
        for i in range(6):
            x = torch.relu(self.tconvs[i](x))
            # x = torch.sigmoid(self.tconvs[i](x))

        # output: (1, 1, 64, 64)
        # return torch.sigmoid(x)
        return x

class Test(nn.Module):
    def __init__(self, frames):
        super(Test, self).__init__()
        # self.encoder = Encoder()
        self.loss = nn.MSELoss()
        self.decoder = Decoder()
        self.N = frames

    def forward(self, states):
        decoded = []
        for frame in range(self.N):
            # encoded = self.encoder(obs[frame][None, None, :]) #tensor(64,)
            d = self.decoder(states[frame]).squeeze()
            decoded.append(d)
        decoded = torch.stack(decoded, dim=0)
        return decoded

    def criterion(self, reward_predicted, reward_targets):
        reward_predicted = reward_predicted.squeeze()
        assert reward_predicted.shape == reward_targets.shape
        return self.loss(reward_predicted, reward_targets)
