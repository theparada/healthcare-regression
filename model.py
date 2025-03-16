from torch import nn

class Regression(nn.Module):
    def __init__(self, inputs, nodes=32):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(inputs, nodes),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.Linear(nodes, 1)
        )

    def forward(self, x):
        output = self.stack(x)
        return output
