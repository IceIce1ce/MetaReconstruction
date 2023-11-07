from .future_memory import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length=5):
        super(Encoder, self).__init__()
        def BasicBlock1(in_channel, out_channel):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False),
                                       torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False))
        
        def BasicBlock2(in_channel, out_channel):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False),
                                       torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1))

        self.n_channel = 3
        self.moduleConv1 = BasicBlock1(self.n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv2 = BasicBlock1(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv3 = BasicBlock1(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.moduleConv4 = BasicBlock2(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)
        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)
        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)
        tensorConv4 = self.moduleConv4(tensorPool3)
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        def BasicBlock(in_channel, out_channel):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False),
                                       torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False))
        
        def Gen(in_channel, out_channel, nc):
            return torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channel, out_channels=nc, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(nc),
                                       torch.nn.ReLU(inplace=False),
                                       torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                                       torch.nn.BatchNorm2d(nc),
                                       torch.nn.ReLU(inplace=False),
                                       torch.nn.Conv2d(in_channels=nc, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                                       torch.nn.Tanh())
        
        def Upsample(in_channel, out_channel):
            return torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       torch.nn.BatchNorm2d(out_channel),
                                       torch.nn.ReLU(inplace=False))

        self.n_channel = 3
        self.moduleConv = BasicBlock(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)
        self.moduleDeconv3 = BasicBlock(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)
        self.moduleDeconv2 = BasicBlock(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)
        self.moduleDeconv1 = Gen(128, self.n_channel, 64)
        
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x) # [4, 512, 32, 32]
        tensorUpsample4 = self.moduleUpsample4(tensorConv) # [4, 256, 64, 64]
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1) # [4, 512, 64, 64]
        tensorDeconv3 = self.moduleDeconv3(cat4) # [4, 256, 64, 64]
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3) # [4, 128, 128, 128]
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1) # [4, 256, 128, 128]
        tensorDeconv2 = self.moduleDeconv2(cat3) # [4, 128, 128, 128]
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2) # [4, 64, 256, 256]
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1) # [4, 128, 256, 256]
        output = self.moduleDeconv1(cat2) # [4, 3, 256, 256]
        return output

class p_mnad(torch.nn.Module):
    def __init__(self, t_length=5, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(p_mnad, self).__init__()
        self.encoder = Encoder(t_length)
        self.decoder = Decoder()
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):
        fea, skip1, skip2, skip3 = self.encoder(x)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss