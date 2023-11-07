from .memory import Memory
import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        def BasicBlock1(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False))

        def BasicBlock2(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1))

        self.moduleConv1 = BasicBlock1(3, 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv2 = BasicBlock1(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv3 = BasicBlock1(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv4 = BasicBlock2(256, 512)

    def forward(self, x):
        x = self.moduleConv1(x)
        x = self.modulePool1(x)
        x = self.moduleConv2(x)
        x = self.modulePool2(x)
        x = self.moduleConv3(x)
        x = self.modulePool3(x)
        x = self.moduleConv4(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        def BasicBlock1(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False))

        def BasicBlock2(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh())

        def Upsample(in_channel, out_channel):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(inplace=False))

        self.moduleConv = BasicBlock1(1024, 512)
        self.moduleUpsample4 = Upsample(512, 512)
        self.moduleDeconv3 = BasicBlock1(512, 256)
        self.moduleUpsample3 = Upsample(256, 256)
        self.moduleDeconv2 = BasicBlock1(256, 128)
        self.moduleUpsample2 = Upsample(128, 128)
        self.moduleDeconv1 = BasicBlock2(128, 64)

    def forward(self, x):
        x = self.moduleConv(x)
        x = self.moduleUpsample4(x)
        x = self.moduleDeconv3(x)
        x = self.moduleUpsample3(x)
        x = self.moduleDeconv2(x)
        x = self.moduleUpsample2(x)
        x = self.moduleDeconv1(x)
        return x

class r_mnad(torch.nn.Module):
    def __init__(self, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(r_mnad, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):
        # x -> [4, 3, 256, 256]
        fea = self.encoder(x)  # [4, 512, 32, 32]
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)  # [4, 3, 256, 256]
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss