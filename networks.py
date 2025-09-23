# Importing necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLD = 0.5

# class UNET(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder (Image Processing)
#         self.conv11 = nn.Conv2d(2, 64, 3, padding=1)
#         self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2)
        
#         self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2)
        
#         self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2)

#         # Traction Processing (2x36x36 input)
#         self.convtrac1 = nn.Conv2d(2, 10, 3, padding=1)
#         self.convtrac2 = nn.Conv2d(10, 20, 3, padding=1)

#         # Bridge
#         self.conv41 = nn.Conv2d(276, 512, 3, padding=1)  # 256 + 20 = 276
#         self.conv42 = nn.Conv2d(512, 512, 3, padding=1)
#         self.drop4 = nn.Dropout(0.5)
#         self.pool4 = nn.MaxPool2d(2)
        
#         self.conv51 = nn.Conv2d(512, 1024, 3, padding=1)
#         self.conv52 = nn.Conv2d(1024, 1024, 3, padding=1)
#         self.drop5 = nn.Dropout(0.5)

#         # Decoder
#         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv61 = nn.Conv2d(1024, 512, 3, padding=1)
#         self.conv62 = nn.Conv2d(512, 512, 3, padding=1)
        
#         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv71 = nn.Conv2d(512, 256, 3, padding=1)
#         self.conv72 = nn.Conv2d(256, 256, 3, padding=1)
        
#         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv81 = nn.Conv2d(256, 128, 3, padding=1)
#         self.conv82 = nn.Conv2d(128, 128, 3, padding=1)
        
#         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv91 = nn.Conv2d(128, 64, 3, padding=1)
#         self.conv92 = nn.Conv2d(64, 64, 3, padding=1)

#         # Output
#         self.conv101 = nn.Conv2d(64, 2, 3, padding=1)
#         self.conv102 = nn.Conv2d(2, 1, 1)
        
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         img, trac = x  # img: (B, 2, 288, 288), trac: (B, 2, 36, 36)

#         # Encoder
#         c1 = self.relu(self.conv11(img))
#         c1 = self.relu(self.conv12(c1))
#         p1 = self.pool1(c1)
        
#         c2 = self.relu(self.conv21(p1))
#         c2 = self.relu(self.conv22(c2))
#         p2 = self.pool2(c2)
        
#         c3 = self.relu(self.conv31(p2))
#         c3 = self.relu(self.conv32(c3))
#         p3 = self.pool3(c3)

#         # Traction Processing
#         t = self.relu(self.convtrac1(trac))
#         t = self.relu(self.convtrac2(t))
#         p3 = torch.cat([p3, t], dim=1)  # (B, 256+20=276, 36, 36)

#         # Bridge
#         c4 = self.relu(self.conv41(p3))
#         c4 = self.relu(self.conv42(c4))
#         d4 = self.drop4(c4)
#         p4 = self.pool4(d4)
        
#         c5 = self.relu(self.conv51(p4))
#         c5 = self.relu(self.conv52(c5))
#         d5 = self.drop5(c5)

#         # Decoder
#         u6 = self.up6(d5)
#         u6 = torch.cat([u6, d4], 1)
#         c6 = self.relu(self.conv61(u6))
#         c6 = self.relu(self.conv62(c6))
        
#         u7 = self.up7(c6)
#         u7 = torch.cat([u7, c3], 1)
#         c7 = self.relu(self.conv71(u7))
#         c7 = self.relu(self.conv72(c7))
        
#         u8 = self.up8(c7)
#         u8 = torch.cat([u8, c2], 1)
#         c8 = self.relu(self.conv81(u8))
#         c8 = self.relu(self.conv82(c8))
        
#         u9 = self.up9(c8)
#         u9 = torch.cat([u9, c1], 1)
#         c9 = self.relu(self.conv91(u9))
#         c9 = self.relu(self.conv92(c9))

#         # Output
#         c10 = self.conv101(c9)
#         c11 = self.conv102(c10)
#         return torch.squeeze(c11, 1)  # Remove channel dim (B, 1, H, W) -> (B, H, W)
# import torch
# import torch.nn as nn

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ------------------ Encoder ------------------
        # Block 1
        self.conv11 = nn.Conv2d(2, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)

        # ------------------ Traction Processor ------------------
        self.convtrac1 = nn.Conv2d(2, 64, 3, padding=1)
        self.bn_trac1 = nn.BatchNorm2d(64)
        self.convtrac2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_trac2 = nn.BatchNorm2d(128)

        # ------------------ Bridge ------------------
        self.conv41 = nn.Conv2d(256+128, 512, 3, padding=1)  # 256 (img) + 128 (trac)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.drop4 = nn.Dropout(0.5)
        
        # ------------------ Bottleneck ------------------
        self.conv51 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn51 = nn.BatchNorm2d(1024)
        self.conv52 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn52 = nn.BatchNorm2d(1024)
        self.drop5 = nn.Dropout(0.5)

        # ------------------ Decoder ------------------
        # Up 1
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv61 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn61 = nn.BatchNorm2d(512)
        self.conv62 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn62 = nn.BatchNorm2d(512)
        
        # Up 2
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv71 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn72 = nn.BatchNorm2d(256)
        
        # Up 3
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv81 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn81 = nn.BatchNorm2d(128)
        self.conv82 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn82 = nn.BatchNorm2d(128)
        
        # Up 4
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv91 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn91 = nn.BatchNorm2d(64)
        self.conv92 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn92 = nn.BatchNorm2d(64)

        # ------------------ Output ------------------
        self.conv101 = nn.Conv2d(64, 2, 3, padding=1)
        self.conv102 = nn.Conv2d(2, 1, 1)
        
        self.relu = nn.ReLU()

    def forward(self, img, trac):
        # img, trac = x  # img: (B, 2, 288, 288), trac: (B, 2, 36, 36)

        # ----------- Encoder -----------
        # Block 1
        c1 = self.relu(self.bn11(self.conv11(img)))
        c1 = self.relu(self.bn12(self.conv12(c1)))
        p1 = self.pool1(c1)
        
        # Block 2
        c2 = self.relu(self.bn21(self.conv21(p1)))
        c2 = self.relu(self.bn22(self.conv22(c2)))
        p2 = self.pool2(c2)
        
        # Block 3
        c3 = self.relu(self.bn31(self.conv31(p2)))
        c3 = self.relu(self.bn32(self.conv32(c3)))
        p3 = self.pool3(c3)

        # ----------- Traction Processing -----------
        t = self.relu(self.bn_trac1(self.convtrac1(trac)))
        t = self.relu(self.bn_trac2(self.convtrac2(t)))
        p3 = torch.cat([p3, t], dim=1)  # (256 + 128 = 384 channels)

        # ----------- Bridge -----------
        c4 = self.relu(self.bn41(self.conv41(p3)))
        c4 = self.relu(self.bn42(self.conv42(c4)))
        d4 = self.drop4(c4)
        p4 = F.max_pool2d(d4, kernel_size=2, stride=2)
        
        # ----------- Bottleneck -----------
        c5 = self.relu(self.bn51(self.conv51(p4)))
        c5 = self.relu(self.bn52(self.conv52(c5)))
        d5 = self.drop5(c5)

        # ----------- Decoder -----------
        # Up 1
        u6 = self.up6(d5)
        u6 = torch.cat([u6, d4], dim=1)
        c6 = self.relu(self.bn61(self.conv61(u6)))
        c6 = self.relu(self.bn62(self.conv62(c6)))
        
        # Up 2
        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.relu(self.bn71(self.conv71(u7)))
        c7 = self.relu(self.bn72(self.conv72(c7)))
        
        # Up 3
        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.relu(self.bn81(self.conv81(u8)))
        c8 = self.relu(self.bn82(self.conv82(c8)))
        
        # Up 4
        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.relu(self.bn91(self.conv91(u9)))
        c9 = self.relu(self.bn92(self.conv92(c9)))

        # ----------- Output -----------
        c10 = self.conv101(c9)
        c11 = self.conv102(c10)
        return torch.squeeze(c11, 1)

# Example usage
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNET().to(device)
    
#     # Example input (batch_size=2)
#     img = torch.randn(2, 2, 288, 288).to(device)
#     trac = torch.randn(2, 2, 36, 36).to(device)
    
#     output = model((img, trac))
#     print(f"Output shape: {output.shape}")  # Should be (2, 288, 288)

# class CNN_MID_INJECT(nn.Module):
#     """
#     A simple convolutional neural network for MNIST classification to test crossbar convolution.
#     """

#     def __init__(self):
#         super().__init__()
#         self.relu = nn.ReLU()

#         self.conv1 = nn.Conv2d(
#             in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(
#             in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(
#             in_channels=4, out_channels=6, kernel_size=3, stride=1, padding=1
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv4 = nn.Conv2d(
#             in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1
#         )

#         self.convt1 = nn.ConvTranspose2d(
#             in_channels=8, out_channels=4, kernel_size=2, stride=2
#         )
#         self.conv5 = nn.Conv2d(
#             in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1
#         )
#         self.conv6 = nn.Conv2d(
#             in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1
#         )

#         self.convt2 = nn.ConvTranspose2d(
#             in_channels=4, out_channels=2, kernel_size=2, stride=2
#         )
#         self.conv7 = nn.Conv2d(
#             in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1
#         )
#         self.conv8 = nn.Conv2d(
#             in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1
#         )

#         self.convt3 = nn.ConvTranspose2d(
#             in_channels=2, out_channels=1, kernel_size=2, stride=2
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         (img, trac) = x

#         x = self.conv1(img)
#         x = self.pool1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.pool3(x)
#         x = self.relu(x)

#         x = torch.cat((x, trac), 1)

#         x = self.conv4(x)

#         x = self.convt1(x)
#         x = self.relu(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.conv6(x)
#         x = self.relu(x)

#         x = self.convt2(x)
#         x = self.relu(x)
#         x = self.conv7(x)
#         x = self.relu(x)
#         x = self.conv8(x)
#         x = self.relu(x)

#         x = self.convt3(x)
#         x = self.sigmoid(x)
#         x = torch.squeeze(x)

#         return x


# class UNET(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv11 = nn.Conv2d(
#             in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1
#         )
#         self.conv12 = nn.Conv2d(
#             in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

#         self.conv21 = nn.Conv2d(
#             in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
#         )
#         self.conv22 = nn.Conv2d(
#             in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

#         self.conv31 = nn.Conv2d(
#             in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
#         )
#         self.conv32 = nn.Conv2d(
#             in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

#         # traction processing layers
#         self.convtrac1 = nn.Conv2d(
#             in_channels=2, out_channels=10, kernel_size=3, stride=1, padding=1
#         )
#         self.convtrac2 = nn.Conv2d(
#             in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1
#         )

#         self.conv41 = nn.Conv2d(
#             in_channels=276, out_channels=512, kernel_size=3, stride=1, padding=1
#         )
#         self.conv42 = nn.Conv2d(
#             in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
#         )
#         self.drop4 = nn.Dropout(0.5)
#         self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

#         self.conv51 = nn.Conv2d(
#             in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
#         )
#         self.conv52 = nn.Conv2d(
#             in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1
#         )
#         self.drop5 = nn.Dropout(0.5)

#         self.up6 = nn.ConvTranspose2d(
#             in_channels=1024, out_channels=512, kernel_size=2, stride=2
#         )
#         self.conv61 = nn.Conv2d(
#             in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1
#         )
#         self.conv62 = nn.Conv2d(
#             in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
#         )

#         self.up7 = nn.ConvTranspose2d(
#             in_channels=512, out_channels=256, kernel_size=2, stride=2
#         )
#         self.conv71 = nn.Conv2d(
#             in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
#         )
#         self.conv72 = nn.Conv2d(
#             in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
#         )

#         self.up8 = nn.ConvTranspose2d(
#             in_channels=256, out_channels=128, kernel_size=2, stride=2
#         )
#         self.conv81 = nn.Conv2d(
#             in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
#         )
#         self.conv82 = nn.Conv2d(
#             in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
#         )

#         self.up9 = nn.ConvTranspose2d(
#             in_channels=128, out_channels=64, kernel_size=2, stride=2
#         )
#         self.conv91 = nn.Conv2d(
#             in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
#         )
#         self.conv92 = nn.Conv2d(
#             in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         )

#         self.conv101 = nn.Conv2d(
#             in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1
#         )
#         self.conv102 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         (img, trac) = x

#         c1 = self.conv11(img)
#         c1 = self.relu(c1)
#         c1 = self.conv12(c1)
#         c1 = self.relu(c1)
#         p1 = self.pool1(c1)

#         c2 = self.conv21(p1)
#         c2 = self.relu(c2)
#         c2 = self.conv22(c2)
#         c2 = self.relu(c2)
#         p2 = self.pool2(c2)

#         c3 = self.conv31(p2)
#         c3 = self.relu(c3)
#         c3 = self.conv32(c3)
#         c3 = self.relu(c3)
#         p3 = self.pool3(c3)

#         t1 = self.convtrac1(trac)
#         t2 = self.convtrac2(t1)
#         # print(trac.size(), t1.size(), t2.size(), p3.size())
#         p3 = torch.cat((p3, t2), dim=1)

#         c4 = self.conv41(p3)
#         c4 = self.relu(c4)
#         c4 = self.conv42(c4)
#         c4 = self.relu(c4)
#         d4 = self.drop4(c4)
#         p4 = self.pool4(d4)

#         c5 = self.conv51(p4)
#         c5 = self.relu(c5)
#         c5 = self.conv52(c5)
#         c5 = self.relu(c5)
#         d5 = self.drop5(c5)

#         u6 = self.up6(d5)
#         u6 = torch.cat((u6, d4), dim=1)
#         c6 = self.conv61(u6)
#         c6 = self.relu(c6)
#         c6 = self.conv62(c6)
#         c6 = self.relu(c6)

#         u7 = self.up7(c6)
#         u7 = torch.cat((u7, c3), dim=1)
#         c7 = self.conv71(u7)
#         c7 = self.relu(c7)
#         c7 = self.conv72(c7)
#         c7 = self.relu(c7)

#         u8 = self.up8(c7)
#         u8 = torch.cat((u8, c2), dim=1)
#         c8 = self.conv81(u8)
#         c8 = self.relu(c8)
#         c8 = self.conv82(c8)
#         c8 = self.relu(c8)

#         u9 = self.up9(c8)
#         u9 = torch.cat((u9, c1), dim=1)
#         c9 = self.conv91(u9)
#         c9 = self.relu(c9)
#         c9 = self.conv92(c9)
#         c9 = self.relu(c9)

#         c10 = self.conv101(c9)
#         c11 = self.conv102(c10)

#         x = torch.squeeze(torch.squeeze(c11))
        
#         return x

if __name__ == "__main__":
    # Example usage with torchviz visualization
    import torch
    from torchviz import make_dot
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET().to(device)
    
    # Create example inputs
    img = torch.randn(1, 2, 288, 288).to(device)
    trac = torch.randn(1, 2, 36, 36).to(device)
    
    # Forward pass
    output = model(img, trac)
    
    # Create visualization
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("unet_architecture", format="png")
    print("Model architecture visualization saved as unet_architecture.png")
