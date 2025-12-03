import matplotlib.pyplot as plt
import numpy as np

def draw_calculation_logic():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 定义关键点
    C = np.array([0, 0])       # 中心点 (Red Dot)
    H = np.array([0, 40])      # 蓝点手柄 (Handle), 距离中心 40
    P = np.array([30, 60])     # 像素点 P (在清晰线外侧)
    
    # 法向量 (垂直向上)
    nx, ny = 0, 1
    
    # 2. 画出三条关键线
    # A. 中心线 (Center Line)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Center Line (Parallel to the Limit)')
    
    # B. 清晰区边界 (Sharp Limit / Dashed Line)
    ax.axhline(40, color='blue', linestyle='-', linewidth=2, label='Sharp Limit')
    
    # C. 像素点的投影线
    ax.plot([P[0], P[0]], [0, P[1]], 'k:', alpha=0.3) # 垂线
    
    # 3. 画出向量和距离标注
    
    # -> 向量 CP (从中心到像素)
    ax.annotate("", xy=P, xytext=C, arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(15, 30, "Distance from center to pixel", color="gray", rotation=60, ha='right')
    
    # -> 距离 1: dist_field (总投影距离)
    # 画在左侧，表示从中心一直到 P 的垂直高度
    ax.annotate("", xy=(-10, 60), xytext=(-10, 0), arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax.text(-12, 30, "2. Point Dist\n(Projection from Center on Normal)", color="green", ha='right', va='center', fontweight='bold')
    
    # -> 距离 2: s_sharp (清晰区半径)
    # 画在右侧，表示从中心到蓝点的距离
    ax.annotate("", xy=(10, 40), xytext=(10, 0), arrowprops=dict(arrowstyle="<->", color="blue", lw=2))
    ax.text(12, 20, "1. Limit Dist\n(Distance to Limit)", color="blue", ha='left', va='center', fontweight='bold')
    
    # -> 距离 3: Result (最终算出的距离)
    # 画在更右侧，表示 P 到蓝线的距离
    ax.annotate("", xy=(40, 60), xytext=(40, 40), arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    ax.text(42, 50, "3. Distance to compute Decay\n= Point Dist - Limit Dist", color="purple", ha='left', va='center', fontweight='bold')
    
    # 4. 画出点
    ax.plot(C[0], C[1], 'ro', markersize=10, label='Center (C)')
    ax.plot(H[0], H[1], 'bo', markersize=10, label='Handle (H)')
    ax.plot(P[0], P[1], 'go', markersize=10, label='Pixel (P)')

    # 设置图表范围和样式
    ax.set_xlim(-20, 60)
    ax.set_ylim(-10, 80)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    ax.set_title("Visualizing the Projection", fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_calculation_logic()
