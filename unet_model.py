import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        
        self.forward_pass = nn.ModuleList()
        self.backward_pass = nn.ModuleList()
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.forward_pass.append(DoubleConvBlock(input_channels, feature))
            input_channels = feature
        
        for feature in reversed(features):
            self.backward_pass.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.backward_pass.append(DoubleConvBlock(feature * 2, feature))
        
        self.bottleneck = DoubleConvBlock(features[-1], features[-1] * 2)
        self.output_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for conv in self.forward_pass:
            x = conv(x)
            skip_connections.append(x)
            x = self.max_pooling(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.backward_pass) // 2):
            x = self.backward_pass[2 * idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.backward_pass[2 * idx + 1](concat_skip)
            
        return self.output_conv(x)
        
def test():
    x = torch.randn((3, 1, 160, 160))
    model = Unet(input_channels=1, output_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape
    
    
if __name__ == '__main__':
    test()
