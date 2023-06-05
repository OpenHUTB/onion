import torch
torch.manual_seed(123)
import torch.nn as nn

class C1D(nn.Module):
    def __init__(self, hparams):
        super(C1D, self).__init__()

        self._extractor = nn.Sequential(
            #first conv
            nn.Conv1d(in_channels=hparams.num_mels, out_channels=64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(3,3),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3,3),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3,3))

        self._classifier = nn.Sequential(nn.Linear(in_features=2240, out_features=2048),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=2048, out_features=1024),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=1024, out_features=len(hparams.genres)))
        self.apply(self._init_weights)
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self._extractor(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self._classifier(x)
        return x

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)