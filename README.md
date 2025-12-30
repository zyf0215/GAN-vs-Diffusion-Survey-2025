# GAN-vs-Diffusion-Survey-2025

# fid_evolution_plot.py                  # FID历史曲线
作者：张一凡，2025年12月
功能：绘制GAN与扩散模型在ImageNet 256x256上的FID分数演进曲线

import matplotlib.pyplot as plt
import numpy as np

# 数据来源：文献统计与2025年最新成果
years = [2014, 2018, 2020, 2021, 2023, 2025]
gan_fid = [None, 20.4, 18.6, 6.81, 4.23, 3.12]      # GAN FID演进
diffusion_fid = [None, None, 9.46, 2.97, 1.97, 1.79]  # 扩散模型FID演进

plt.figure(figsize=(10, 6))
plt.plot(years, gan_fid, 'go-', label='GAN FID (Lower is Better)', linewidth=2, markersize=8)
plt.plot(years[2:], diffusion_fid[2:], 'r^-', label='Diffusion Model FID', linewidth=2, markersize=8)

# 填充技术差距区域
plt.fill_between(years[2:], gan_fid[2:], diffusion_fid[2:], color='gray', alpha=0.3, label='Technology Gap')

plt.title('Evolution of FID Scores: GANs vs Diffusion Models (ImageNet 256×256)\n2014–2025', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('FID Score ↓', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xticks(years)
plt.tight_layout()

# 保存高清图片用于报告插入
plt.savefig('visual_results/fid_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("FID演进曲线已保存至 visual_results/fid_curve.png")

# inference_speed_bar_chart.py           # 推理速度对比柱状图
# 功能：2025年主流模型推理速度对比（单张A100 GPU）

import matplotlib.pyplot as plt

models = ['StyleGAN3', 'SD 1.5', 'SDXL', 'SD3-Medium', 'GAD (2025)', 'Rectified Flow v2']
steps = [1, 50, 30, 28, 8, 1]
times = [0.03, 3.8, 2.8, 2.1, 0.42, 0.19]  # 秒

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Inference Steps', color=color, fontsize=12)
ax1.bar(models, steps, color=color, alpha=0.7, label='Inference Steps')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time per Image (seconds) ↓', color=color, fontsize=12)
ax2.plot(models, times, 'ro--', linewidth=2, markersize=8, label='Time (s)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Inference Efficiency Comparison (2025 Benchmarks)\nLower Time & Fewer Steps = Better', 
          fontsize=14, fontweight='bold', pad=20)
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.xticks(rotation=15)
plt.savefig('visual_results/speed_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("推理速度对比图已保存至 visual_results/speed_comparison.png")

# simple_gan_demo.py  简单GAN演示（MNIST）
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

          
# simple_diffusion_demo.ipynb  简单扩散模型演示（使用diffusers库）
# 使用Hugging Face diffusers库加载Stable Diffusion生成图像

!pip install diffusers transformers accelerate  # 如果环境未安装

from diffusers import StableDiffusionPipeline
import torch

# 加载预训练模型（首次运行会下载）
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 生成图像
prompt = "a beautiful sunset over mountains, highly detailed, 4k"
image = pipe(prompt).images[0]

# 显示并保存
image.save("visual_results/diffusion_generated_sunset.png")
image
#requiremnts.txt
torch>=2.0.0
torchvision
matplotlib
numpy
diffusers
transformers
accelerate

