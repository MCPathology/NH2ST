import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os


def load_data(file_path):
    """加载数据文件"""
    data = np.load(file_path)
    gene_expression = np.atleast_1d(data['gene_expression'])  # 将零维数组转换为一维数组
    predicted_expression = np.atleast_1d(data['predicted_expression'])
    position = np.atleast_2d(data['position'])
    return gene_expression, predicted_expression, position


def plot_expression(expression, positions, title, output_path, cmap='viridis'):
    """绘制基因表达图"""
    plt.figure(figsize=(6, 6))
    norm = Normalize(vmin=np.min(expression), vmax=np.max(expression))
    sc = plt.scatter(positions[:, 0], positions[:, 1], c=expression, cmap=cmap, norm=norm, s=250, edgecolor='none')
    plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=30)
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def plot_gene_expression_files(directory):
    """绘制指定目录下的所有基因表达文件"""
    gene_expressions = []
    predicted_expressions = []
    positions = []

    for file in os.listdir(directory):
        if file.endswith('.npz'):
            gene_expr, pred_expr, pos = load_data(os.path.join(directory, file))
            gene_expressions.append(gene_expr)
            predicted_expressions.append(pred_expr)
            positions.append(pos)
            print("pos", pos)

    # 将列表转换为numpy数组
    gene_expressions = np.concatenate(gene_expressions)
    predicted_expressions = np.concatenate(predicted_expressions)
    positions = np.concatenate(positions)

    plot_expression(gene_expressions, positions, 'Gene Expression',
                    os.path.join(directory, 'combined_gene_expression.png'))
    plot_expression(predicted_expressions, positions, 'Predicted Expression',
                    os.path.join(directory, 'combined_predicted_expression.png'))


# 示例用法
directory = "final_ST/Human_Heart_Asp_12122019_ST_ST_Sample_6.5PCW_1+STimage-1K4M"
plot_gene_expression_files(directory)