# fid_evolution_plot.py
# 作者：zyf0215，2025年12月
# 功能：绘制GAN与扩散模型在ImageNet 256x256上的FID分数演进曲线

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
