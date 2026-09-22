from torch import nn


HIDDEN_LAYERS = (256, 256, 128, 128, 64, 64, 32, 32)

class MLP(nn.Module):
    def __init__(self, input_dim, num_outputs, activation_factory, hidden_layers=HIDDEN_LAYERS, dropout=0.2):
        super().__init__()
        layers, previous = [], input_dim

        for width in hidden_layers:
            layers.extend((nn.Linear(previous, width), activation_factory(width)))
            if dropout:
                layers.append(nn.Dropout(dropout))

            previous = width

        self.features, self.output = nn.Sequential(*layers), nn.Linear(previous, num_outputs)

    def forward(self, x):
        return self.output(self.features(x))
