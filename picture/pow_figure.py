import numpy as np
import matplotlib.pyplot as plt

# 创建x值的数组
x = np.linspace(0, 1, 100)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(3, 3))

# 绘制多条pow曲线
powers = [.3, .4, 0.5]
for power in powers:
    y = np.power(x, power)
    ax.plot(x, y, label=f'x^{power}')

# 设置图形标题和标签
# ax.set_title('Multiple Power Curves')
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# 添加图例
# ax.legend()

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 保存为SVG格式
plt.savefig('power_curves.svg', format='svg', dpi=300, bbox_inches='tight')

print("SVG文件已保存为 'power_curves.svg'")