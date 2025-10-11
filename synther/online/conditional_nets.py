import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

class Curiosity(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Curiosity, self).__init__()

        self.resnet_time = 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_size + self.hidden_size, self.hidden_size),
                Swish(),
                nn.Linear(self.hidden_size, self.hidden_size),
                )] * 2 * self.resnet_time
        )

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.output_size + self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.output_size + self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size),
            Swish(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state, next_state, action):
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state

        return real_next_state_feature, pred_next_state_feature, pred_action

    def compute_reward(self, state, next_state, action, reward, done):
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)

        real_next_state_feature, pred_next_state_feature, _ = self.forward(state, next_state, action)
        icm_reward = F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(1, keepdim=True)

        return icm_reward
    
    def compute_reward_torch(self, state, next_state, action):
        real_next_state_feature, pred_next_state_feature, _ = self.forward(state, next_state, action)
        icm_reward = F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(1, keepdim=True)

        return icm_reward

    def forward_loss(self, state, next_state, action):
        real_next_state_feature, pred_next_state_feature, pred_action = self.forward(state, next_state, action)
        loss = F.mse_loss(real_next_state_feature.detach(), pred_next_state_feature) + F.mse_loss(action, pred_action)
        return loss


class Predictor(nn.Module):
    def __init__(self, input_size, output_size=8, num_hidden=4, hidden_size=32, init_w = 3e-3, b_init_value=0.1, normalize=False, return_preactivation=False):
        super().__init__()
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_activation = F.relu
        self.output_activation = nn.Identity()
        self.normalize = normalize
        in_size = input_size
        
        self.fcs = []
        for i in range(num_hidden):
            fc = nn.Linear(in_size, hidden_size)
            in_size = hidden_size
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)
            self.add_module("fc{}".format(i), fc)
            
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, input, returen_preactivations=False):
        if self.normalize:
            mean_input = input.mean(0, keepdim=True)
            std_input = input.std(0, keepdim=True) + 1e-6
            input = (input - mean_input) / std_input
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if returen_preactivations:
            return output, preactivation
        else:
            return output
        
