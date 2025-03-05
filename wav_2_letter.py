from torch import nn
from audio_data import base_channels

class wav2letter(nn.Module):
    def __init__(self):
        
        super(wav2letter, self).__init__()
        self.conv1 = nn.Conv1d(base_channels * 3, 250, 48, 2, padding = 23)
        self.conv2 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv3 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv4 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv5 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv6 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv7 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv8 = nn.Conv1d(250, 250, 7, padding = 3)
        self.conv9 = nn.Conv1d(250, 2000, 32, padding = 15)
        self.conv10 = nn.Conv1d(2000, 2000, 1)
        self.conv11 = nn.Conv1d(2000, 40, 1)
        self.conv12 = nn.Conv1d(40, 29, 1)
        

    def forward(self, x):
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.conv7(x))
        x = nn.functional.relu(self.conv8(x))
        x = nn.functional.relu(self.conv9(x))
        x = nn.functional.relu(self.conv10(x))
        x = nn.functional.relu(self.conv11(x))
        x = nn.functional.log_softmax(self.conv12(x), dim = 1)
        
        return x