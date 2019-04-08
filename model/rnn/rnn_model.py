import torch
from torch import nn


class RadarRnn1(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RadarRnn1, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=4,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)
        self.out = nn.Sequential(nn.Softmax())  # 分类器，预测位置最大的一个

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.fc(r_out[:, time, :]))
        out = torch.stack(outs, dim=1)
        b, s, h = r_out.shape  # (batch,seq, , hidden)
        x = out.view(b, s)  # 转化为线性层的输入方式
        return self.out(x), h_state


class RadarRnn2(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RadarRnn2, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=4,
            batch_first=True,
            # bidirectional=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


class RadarRnn3(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RadarRnn3, self).__init__()

        self.rnn1 = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=8,
            batch_first=True
        )

        self.fc1 = nn.Linear(32, 1)

        self.rnn2 = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=8,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn1(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.fc1(r_out[:, time, :]))

        x = torch.stack(outs, dim=1)

        r_out, h_state = self.rnn2(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))

        # out =torch.sigmoid(torch.stack(outs, dim=1)) > 0.5
        # out = torch.eq(out, True).cuda(0)
        # out = out.float()
        return torch.stack(outs, dim=1), h_state


class RadarRnn4(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RadarRnn4, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=8,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


class RadarRnn5(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RadarRnn5, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=8,
            batch_first=True
        )

        self.out = nn.Linear(64, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state
