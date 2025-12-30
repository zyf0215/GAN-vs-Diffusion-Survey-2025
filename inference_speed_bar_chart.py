# inference_speed_bar_chart.py
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
