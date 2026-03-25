"""
NLP实验：RAG + 相似度对比 + 可视化
1. 设定简单词汇库
2. 向量表示（手动定义 + embedding模型）
3. 计算不同距离度量
4. matplotlib可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 设定简单词汇库
# ============================================

# 分类词汇库
vocabulary = {
    "动物": ["猫", "狗", "狮子", "老虎", "鸟", "鱼"],
    "水果": ["苹果", "香蕉", "橙子", "葡萄", "西瓜"],
    "交通": ["汽车", "火车", "飞机", "轮船"],
    "颜色": ["红色", "蓝色", "绿色", "黄色"]
}

# 所有词汇列表
all_words = []
for category in vocabulary.values():
    all_words.extend(category)

print("=" * 50)
print("词汇库")
print("=" * 50)
for cat, words in vocabulary.items():
    print(f"{cat}: {', '.join(words)}")
print(f"\n总共 {len(all_words)} 个词汇")


# ============================================
# 2. 向量表示 - 方法一：手动定义特征向量
# ============================================

def create_manual_vectors(words):
    """
    手动定义特征向量
    特征维度: [实体性, 生命性, 有机物性, 移动性, 颜色感知, 人工性]
    - 实体性: 物体的具体/抽象程度
    - 生命性: 是否有生命
    - 有机物性: 是否为有机物/可食用
    - 移动性: 自主移动的能力
    - 颜色感知: 与颜色的关联程度
    - 人工性: 是否为人造物
    """
    # 定义每个词的特征向量
    feature_dict = {
        # 动物 - 高生命性、高移动性、中等实体性、自然性
        "猫":    [0.8, 0.95, 0.0, 0.6, 0.0, 0.0],
        "狗":    [0.8, 0.95, 0.0, 0.7, 0.0, 0.0],
        "狮子":  [0.9, 0.95, 0.0, 0.75, 0.0, 0.0],
        "老虎":  [0.9, 0.95, 0.0, 0.75, 0.0, 0.0],
        "鸟":    [0.6, 0.9, 0.0, 0.95, 0.0, 0.0],
        "鱼":    [0.7, 0.95, 0.0, 0.85, 0.0, 0.0],
        # 水果 - 高有机物性、可食用、低移动性
        "苹果":  [0.75, 0.1, 0.95, 0.0, 0.4, 0.0],
        "香蕉":  [0.7, 0.05, 0.95, 0.0, 0.3, 0.0],
        "橙子":  [0.75, 0.1, 0.9, 0.0, 0.6, 0.0],
        "葡萄":  [0.5, 0.1, 0.9, 0.0, 0.5, 0.0],
        "西瓜":  [0.85, 0.05, 0.95, 0.0, 0.35, 0.0],
        # 交通 - 高移动性、高人工性、中等实体性
        "汽车":  [0.85, 0.0, 0.0, 0.75, 0.0, 0.95],
        "火车":  [0.9, 0.0, 0.0, 0.85, 0.0, 0.95],
        "飞机":  [0.85, 0.0, 0.0, 0.95, 0.0, 0.95],
        "轮船":  [0.85, 0.0, 0.0, 0.65, 0.0, 0.9],
        # 颜色 - 抽象概念、低实体性、高颜色感知
        "红色":  [0.3, 0.3, 0.0, 0.0, 1.0, 0.0],
        "蓝色":  [0.3, 0.0, 0.0, 0.0, 1.0, 0.0],
        "绿色":  [0.3, 0.5, 0.2, 0.0, 1.0, 0.0],
        "黄色":  [0.3, 0.1, 0.3, 0.0, 1.0, 0.0],
    }

    vectors = np.array([feature_dict[word] for word in words])
    return vectors, feature_dict

manual_vectors, feature_dict = create_manual_vectors(all_words)

print("\n" + "=" * 50)
print("方法一：手动定义特征向量")
print("=" * 50)
print(f"特征维度: [实体性, 生命性, 有机物性, 移动性, 颜色感知, 人工性]")
print(f"\n示例 - '狮子' 的特征向量: {feature_dict['狮子']}")


# ============================================
# 2. 向量表示 - 方法二：使用预训练词向量 (模拟)
# ============================================

def create_embedding_vectors(words):
    """
    模拟预训练词向量 (实际项目中可使用 word2vec/glove/bert等)
    这里使用随机种子生成稳定的"伪"embedding
    """
    np.random.seed(42)
    # 生成50维的模拟embedding向量
    dim = 50
    word_to_embedding = {}

    # 为每个类别设置不同的基础向量，使同类词更相似
    category_bases = {
        "动物": np.random.randn(dim) * 0.5 + np.array([1, 0, 0, 0] + [0]*(dim-4)),
        "水果": np.random.randn(dim) * 0.5 + np.array([0, 1, 0, 0] + [0]*(dim-4)),
        "交通": np.random.randn(dim) * 0.5 + np.array([0, 0, 1, 0] + [0]*(dim-4)),
        "颜色": np.random.randn(dim) * 0.5 + np.array([0, 0, 0, 1] + [0]*(dim-4)),
    }

    for cat, cat_words in vocabulary.items():
        for word in cat_words:
            # 类别基础向量 + 词特定偏移
            word_idx = all_words.index(word)
            word_embedding = category_bases[cat] + np.random.randn(dim) * 0.1
            word_to_embedding[word] = word_embedding

    vectors = np.array([word_to_embedding[word] for word in words])
    return vectors, word_to_embedding

embedding_vectors, word_to_embedding = create_embedding_vectors(all_words)

print("\n" + "=" * 50)
print("方法二：模拟Embedding向量 (50维)")
print("=" * 50)
print(f"向量维度: {embedding_vectors.shape[1]}")
print(f"实际项目中可替换为: Word2Vec, GloVe, BERT, OpenAI Embeddings等")


# ============================================
# 3. 计算不同距离度量
# ============================================

def calculate_similarities(vectors, words, method_name):
    """计算并返回相似度矩阵"""
    # 余弦相似度
    cosine_sim = cosine_similarity(vectors)
    # 欧几里得距离
    euclidean_dist = euclidean_distances(vectors)
    # 曼哈顿距离
    manhattan_dist = manhattan_distances(vectors)

    return {
        'cosine': cosine_sim,
        'euclidean': euclidean_dist,
        'manhattan': manhattan_dist
    }

print("\n" + "=" * 50)
print("3. 距离计算结果")
print("=" * 50)

# 计算手动向量的相似度
manual_similarities = calculate_similarities(manual_vectors, all_words, "手动特征向量")

# 计算embedding向量的相似度
embedding_similarities = calculate_similarities(embedding_vectors, all_words, "Embedding向量")

# 展示部分结果
print("\n余弦相似度示例 (手动特征向量):")
print("狮子 vs 老虎:", manual_similarities['cosine'][all_words.index('狮子'), all_words.index('老虎')].round(4))
print("狮子 vs 苹果:", manual_similarities['cosine'][all_words.index('狮子'), all_words.index('苹果')].round(4))
print("猫 vs 狗:", manual_similarities['cosine'][all_words.index('猫'), all_words.index('狗')].round(4))
print("汽车 vs 飞机:", manual_similarities['cosine'][all_words.index('汽车'), all_words.index('飞机')].round(4))


# ============================================
# 4. 可视化
# ============================================

def visualize_word_vectors(vectors, words, title, save_path=None):
    """使用PCA降维并可视化词向量"""
    # PCA降维到2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 10))

    # 为不同类别设置颜色
    category_colors = {
        "动物": 'red',
        "水果": 'green',
        "交通": 'blue',
        "颜色": 'purple'
    }

    # 绘制点和标签
    for i, word in enumerate(words):
        # 找到词汇所属类别
        for cat, cat_words in vocabulary.items():
            if word in cat_words:
                color = category_colors[cat]
                break

        ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                   c=color, s=150, alpha=0.7, edgecolors='black', linewidths=1)
        ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                   fontsize=12, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points')

    # 添加图例
    for cat, color in category_colors.items():
        ax.scatter([], [], c=color, s=150, label=cat, edgecolors='black')
    ax.legend(loc='upper right', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return vectors_2d

def plot_similarity_heatmap(similarity_matrix, words, title, save_path=None):
    """绘制相似度热力图"""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')

    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(words, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_similarity_heatmap_combined(manual_vecs, embed_vecs, words, save_path=None):
    """合并绘制手动向量和Embedding向量的余弦相似度热力图对比"""
    cos_sim_manual = cosine_similarity(manual_vecs)
    cos_sim_embed = cosine_similarity(embed_vecs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 16))

    for ax, sim_matrix, title in [
        (axes[0], cos_sim_manual, "手动特征向量 - 余弦相似度热力图"),
        (axes[1], cos_sim_embed, "Embedding向量 - 余弦相似度热力图")
    ]:
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(words, fontsize=9)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_distance_comparison(vectors, words, title, save_path=None):
    """绘制词汇间的距离连线图"""
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # 计算余弦相似度
    cos_sim = cosine_similarity(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制连线（相似度越高，线越粗越红）
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            sim = cos_sim[i, j]
            if sim > 0.7:
                alpha = (sim - 0.7) / 0.3
                alpha = min(max(alpha, 0.0), 1.0)
                ax.plot([vectors_2d[i, 0], vectors_2d[j, 0]],
                       [vectors_2d[i, 1], vectors_2d[j, 1]],
                       'r-', alpha=alpha, linewidth=(sim-0.7)*10)

    # 绘制点
    category_colors = {"动物": 'red', "水果": 'green', "交通": 'blue', "颜色": 'purple'}

    for i, word in enumerate(words):
        for cat, cat_words in vocabulary.items():
            if word in cat_words:
                color = category_colors[cat]
                break
        ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                   fontsize=12, ha='center', fontweight='bold',
                   xytext=(0, 12), textcoords='offset points')

    # 图例
    for cat, color in category_colors.items():
        ax.scatter([], [], c=color, s=200, label=cat, edgecolors='black')
    ax.legend(loc='upper right', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


print("\n" + "=" * 50)
print("4. 可视化展示")
print("=" * 50)

def plot_combined_comparison(manual_vecs, embed_vecs, words, save_path=None):
    """合并绘制手动向量和Embedding向量的对比图"""
    pca_manual = PCA(n_components=2)
    pca_embed = PCA(n_components=2)
    vectors_2d_manual = pca_manual.fit_transform(manual_vecs)
    vectors_2d_embed = pca_embed.fit_transform(embed_vecs)

    cos_sim_manual = cosine_similarity(manual_vecs)
    cos_sim_embed = cosine_similarity(embed_vecs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 18))
    category_colors = {"动物": 'red', "水果": 'green', "交通": 'blue', "颜色": 'purple'}

    for ax, vectors_2d, cos_sim, title in [
        (axes[0], vectors_2d_manual, cos_sim_manual, "手动特征向量 - 词汇空间分布"),
        (axes[1], vectors_2d_embed, cos_sim_embed, "Embedding向量 - 词汇空间分布")
    ]:
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                sim = cos_sim[i, j]
                if sim > 0.7:
                    alpha = (sim - 0.7) / 0.3
                    alpha = min(max(alpha, 0.0), 1.0)
                    ax.plot([vectors_2d[i, 0], vectors_2d[j, 0]],
                           [vectors_2d[i, 1], vectors_2d[j, 1]],
                           'r-', alpha=alpha, linewidth=(sim-0.7)*10)

        for i, word in enumerate(words):
            for cat, cat_words in vocabulary.items():
                if word in cat_words:
                    color = category_colors[cat]
                    break
            ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                       c=color, s=200, alpha=0.8, edgecolors='black', linewidths=2, zorder=5)
            ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                       fontsize=12, ha='center', fontweight='bold',
                       xytext=(0, 12), textcoords='offset points')

        for cat, color in category_colors.items():
            ax.scatter([], [], c=color, s=200, label=cat, edgecolors='black')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("\n图1: 手动向量 vs Embedding向量 完整对比图")
plot_combined_comparison(manual_vectors, embedding_vectors, all_words,
                        "manual_vs_embedding_comparison.png")

print("\n图2: 余弦相似度热力图对比")
plot_similarity_heatmap_combined(manual_vectors, embedding_vectors, all_words,
                                  "cosine_similarity_heatmap_comparison.png")


# ============================================
# 5. 简单RAG演示
# ============================================

print("\n" + "=" * 50)
print("5. 简单RAG检索演示")
print("=" * 50)

def simple_rag_retrieval(query_vector, document_vectors, documents, top_k=3):
    """简单的RAG检索：返回与查询最相似的文档"""
    query_vector = query_vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, document_vectors)[0]

    # 获取top-k相似文档
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx]
        })
    return results

# 模拟RAG场景：用"大型猫科动物"查询
print("\n查询: '大型猫科动物'")
# 构建查询向量 (假设有动物性、大型特征)
query_vec = np.array([0.95, 0.0, 0.0, 0.0, 0.85, 0.6])  # 类似狮虎的特征

results = simple_rag_retrieval(query_vec, manual_vectors, all_words, top_k=5)
print("\n检索结果 (Top 5):")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['document']} (相似度: {r['similarity']:.4f})")


print("\n" + "=" * 50)
print("实验完成!")
print("=" * 50)
print("\n生成的图片文件:")
print("  1. manual_vectors_2d.png - 手动特征向量空间分布")
print("  2. cosine_similarity_heatmap.png - 相似度热力图")
print("  3. word_distance_network.png - 词汇关系网络图")
print("  4. embedding_vectors_2d.png - Embedding向量空间分布")
