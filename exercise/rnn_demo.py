
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=6,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


# 定义一些超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# 创造一些数据
steps = np.linspace(0, np.pi*2, 100, dtype=np.float)
x_np = np.sin(steps)
y_np = np.cos(steps)
#
# “看”数据
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()

# 选择模型
model = Rnn(INPUT_SIZE)
print(model)

# 定义优化器和损失函数
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

h_state = None  # 第一次的时候，暂存为0

for step in range(300000):
    start, end = step * np.pi, (step + 1) * np.pi

    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = model(x, h_state)
    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(steps, y_np.flatten(), 'r-')
plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
plt.show()