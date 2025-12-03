from graphviz import Digraph

def draw_algorithm_diagram():
    # 创建一个有向图
    dot = Digraph(comment='Tilt-Shift Algorithm Block Diagram')
    
    # --- 视觉优化设置 ---
    # rankdir='TB': 从上到下布局 (Top to Bottom)，让图变瘦长
    # splines='ortho': 使用折线，看起来更像工程图
    # nodesep/ranksep: 调整节点间距，防止拥挤
    dot.attr(rankdir='TB', size='15,20', dpi='300', splines='ortho', nodesep='0.6', ranksep='0.6')
    dot.attr('node', shape='rect', style='filled', fontname='Helvetica')
    dot.attr('edge', fontsize='10') # 调整连线文字大小

    # --- 1. 输入部分 (内容保持不变) ---
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Inputs', style='dashed', color='grey')
        # 让 Img 和 Settings 在同一水平线上
        c.attr(rank='same')
        c.node('Img', 'Initial Image\n(I_init)', fillcolor='#E1F5FE')
        c.node('Settings', 'Settings\n(Angle, Trans&Blur Limits Distances\n/ Center Position)', fillcolor='#E1F5FE')

    # --- 2. 预处理与模糊 (内容保持不变) ---
    with dot.subgraph(name='cluster_blur') as c:
        c.attr(label='Blur Pipeline', color='blue')
        c.node('BlurAlgo', 'Gaussian Blur', fillcolor='#FFF9C4')
        c.node('ImgBlur', 'Blurred Image\n(I_blur)', fillcolor='#FFCCBC')

    # --- 3. 区域定义 (内容保持不变) ---
    with dot.subgraph(name='cluster_zone') as c:
        c.attr(label='Zone Definition Pipeline', color='green')
        
        c.node('Downscale', 'Downscale Grid (X,Y)', fillcolor='#C8E6C9')
        c.node('Vectors', 'Vector Calculation\n(Normal vector of the limit)', fillcolor='#C8E6C9')
        c.node('DotProd', 'Distance of all points(X,Y)\n(Distance from center to point projected on normal)', fillcolor='#C8E6C9')
        c.node('Zones', 'Trans and Blur Zones \n(Compare point dist and limits)', fillcolor='#FFCCBC')
        
    # --- 4. Mask定义 (内容保持不变) ---
    with dot.subgraph(name='cluster_mask') as c:
        c.attr(label='Mask Pipeline', color='red')
            
        c.node('Decay', 'Decay in Trans Zone\n(Decay Rate = 1-(point dist-Trans lim)/Blur lim)', fillcolor='#C8E6C9')
        c.node('Upscale', 'Upscale Mask\n(Linear Interpolation)', fillcolor='#C8E6C9')
        c.node('Mask', 'Full Mask (M)', fillcolor='#FFCCBC')

    # --- 5. 合成 (内容保持不变) ---
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Compositing', style='dashed')
        c.node('Blend', 'Linear Blending\nI = I_sharp*M + I_blur*(1-M)', shape='diamond', fillcolor='#D1C4E9')
        c.node('Result', 'Final Miniature\nImage', shape='doubleoctagon', fillcolor='#B3E5FC', style='filled,bold')

    # --- 定义连线 (优化版) ---
    # 定义一个辅助函数，强制线条 "下出上进"，保持整齐
    def link(u, v, label=None, style=None):
        dot.edge(u, v, label=label, style=style, tailport='s', headport='n')

    # 1. 模糊流
    link('Img', 'BlurAlgo')
    link('BlurAlgo', 'ImgBlur')
    
    # 2. 核心计算流
    dot.edge('Img', 'Downscale', label='Image shape')
    link('Downscale', 'DotProd', label='Grid (X,Y)')
    
    # Settings 分流 (关键优化点)
    # Settings -> Vectors: 正常向下
    link('Settings', 'Vectors', label='Angle')
    # Settings -> Zones: 跳跃连接，强制从右侧进入，避免穿过中间的框
    #dot.edge('Settings', 'Zones', label='Limits Distances', tailport='e', headport='e', constraint='false')
    
    link('Vectors', 'DotProd', label='normal')
    link('DotProd', 'Zones', label='Point Dist')
    link('Zones', 'Decay', label='Marices of Bool')
    link('Decay', 'Upscale', label='Downscaled Mask')
    link('Upscale', 'Mask')
    
    # 3. 最终合成流 (关键优化点：三路汇聚)
    # 强制分别从 左(w)、右(e)、上(n) 进入 Blend 节点，形成完美的汇聚效果
    dot.edge('ImgBlur', 'Blend', label='I_blur', tailport='s', headport='ne')   # 右路
    dot.edge('Img', 'Blend', label='I_init', tailport='e', headport='ne')       # 左路
    dot.edge('Mask', 'Blend',  label='M', tailport='s', headport='n')                                    # 中路
    
    link('Blend', 'Result')

    # 渲染
    dot.render('tilt_shift_diagram_pretty', view=True, format='png')
    print("Diagram generated: tilt_shift_diagram_pretty.png")

if __name__ == '__main__':
    draw_algorithm_diagram()
