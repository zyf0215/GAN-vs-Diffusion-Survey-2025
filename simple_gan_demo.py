# simple_gan_demo.py
# 一个极简GAN在MNIST上手写数字生成演示

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# 超参数
latent_dim = 100
batch_size = 64
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST('.', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x.view(-1, 784))

G = Generator().to(device)
D = Discriminator().to(device)
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# 训练
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch = real_imgs.size(0)
        
        # 训练判别器
        real_labels = torch.ones(batch, 1).to(device)
        fake_labels = torch.zeros(batch, 1).to(device)
        
        outputs = D(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(batch, latent_dim).to(device)
        fake_imgs = G(z)
        outputs = D(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        outputs = D(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

# 生成样本并保存
z = torch.randn(16, latent_dim).to(device)
generated = G(z).cpu()
save_image(generated, 'visual_results/gan_generated_samples.png', nrow=4, normalize=True)
print("GAN生成样本已保存至 visual_results/gan_generated_samples.png")
