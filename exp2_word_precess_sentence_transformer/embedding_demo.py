"""
Embedding模型对比演示
展示不同Embedding模型在实际词汇上的效果
依赖: pip install sentence-transformers numpy matplotlib scikit-learn
或使用Ollama embedding (需启动ollama服务)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

vocabulary = {
    "动物": ["猫", "狗", "狮子", "老虎", "鸟", "鱼"],
    "水果": ["苹果", "香蕉", "橙子", "葡萄", "西瓜"],
    "交通": ["汽车", "火车", "飞机", "轮船"],
    "颜色": ["红色", "蓝色", "绿色", "黄色"]
}

all_words = []
for category in vocabulary.values():
    all_words.extend(category)


def get_embedding_model():
    """获取可用的embedding模型"""
    models = []

    try:
        from sentence_transformers import SentenceTransformer
        models.append(("sentence-transformers", SentenceTransformer))
    except ImportError:
        pass

    try:
        import ollama
        models.append(("ollama", ollama))
    except ImportError:
        pass

    return models


def embed_with_sentence_transformer(words, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """使用sentence-transformers获取embedding"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(words, show_progress_bar=True)
    return embeddings, model


def embed_with_ollama(words, model_name="nomic-embed-text"):
    """使用Ollama获取embedding"""
    import ollama

    embeddings = []
    for word in words:
        response = ollama.embeddings(model=model_name, prompt=word)
        embeddings.append(response["embedding"])

    return np.array(embeddings), model_name


def calculate_similarities(vectors):
    """计算余弦相似度矩阵"""
    return cosine_similarity(vectors)


def visualize_embeddings(vectors, words, title, save_path=None):
    """使用PCA降维并可视化词向量"""
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))

    category_colors = {
        "动物": 'red',
        "水果": 'green',
        "交通": 'blue',
        "颜色": 'purple'
    }

    for i, word in enumerate(words):
        for cat, cat_words in vocabulary.items():
            if word in cat_words:
                color = category_colors[cat]
                break

        ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                   c=color, s=150, alpha=0.7, edgecolors='black', linewidths=1)
        ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                   fontsize=12, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points')

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

    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(words, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_word_distance_network(vectors, words, title, threshold=0.5, save_path=None):
    """绘制词汇距离连线图"""
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    cos_sim = cosine_similarity(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            sim = cos_sim[i, j]
            if sim > threshold:
                alpha = (sim - threshold) / (1 - threshold)
                alpha = min(max(alpha, 0.0), 1.0)
                ax.plot([vectors_2d[i, 0], vectors_2d[j, 0]],
                       [vectors_2d[i, 1], vectors_2d[j, 1]],
                       'r-', alpha=alpha, linewidth=(sim - threshold) * 8)

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


def analyze_similarity_pairs(vectors, words, top_n=10):
    """分析最相似和最不相似的词对"""
    sim_matrix = cosine_similarity(vectors)

    pairs = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            pairs.append((words[i], words[j], sim_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n最相似的词对 (Top {}):".format(top_n))
    for w1, w2, sim in pairs[:top_n]:
        print(f"  {w1} - {w2}: {sim:.4f}")

    print("\n最不相似的词对 (Bottom {}):".format(top_n))
    for w1, w2, sim in pairs[-top_n:]:
        print(f"  {w1} - {w2}: {sim:.4f}")

    return pairs


def compare_within_category(vectors, words):
    """分析类别内和类别间的平均相似度"""
    sim_matrix = cosine_similarity(vectors)

    category_words = {}
    for cat, cat_words in vocabulary.items():
        category_words[cat] = [words.index(w) for w in cat_words]

    within_category_sims = []
    between_category_sims = []

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            w1_cat = None
            w2_cat = None
            for cat, indices in category_words.items():
                if words[i] in vocabulary[cat]:
                    w1_cat = cat
                if words[j] in vocabulary[cat]:
                    w2_cat = cat

            if w1_cat == w2_cat and w1_cat is not None:
                within_category_sims.append(sim_matrix[i, j])
            else:
                between_category_sims.append(sim_matrix[i, j])

    print("\n类别内平均相似度: {:.4f}".format(np.mean(within_category_sims)))
    print("类别间平均相似度: {:.4f}".format(np.mean(between_category_sims)))
    print("类别内/类别间 比值: {:.4f}".format(np.mean(within_category_sims) / np.mean(between_category_sims)))


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("Embedding模型对比演示")
    print("=" * 60)

    available_models = get_embedding_model()

    if not available_models:
        print("\n错误: 未找到可用的embedding模型。")
        print("请安装以下依赖之一:")
        print("  pip install sentence-transformers")
        print("  或")
        print("  pip install ollama (需启动ollama服务)")
        return

    print("\n可用的embedding模型:")
    for name, _ in available_models:
        print(f"  - {name}")

    model_choice = input("\n请选择模型 (输入序号): ").strip()

    try:
        idx = int(model_choice) - 1
        if idx < 0 or idx >= len(available_models):
            print("无效的选择，使用第一个可用模型")
            idx = 0
    except ValueError:
        idx = 0

    model_type, _ = available_models[idx]

    print(f"\n使用模型: {model_type}")

    if model_type == "sentence-transformers":
        print("\n加载模型 paraphrase-multilingual-MiniLM-L12-v2 (支持中文)...")
        vectors, model_name = embed_with_sentence_transformer(all_words)
        model_name = f"sentence-transformers: {model_name}"
    else:
        print("\n使用 Ollama nomic-embed-text 模型...")
        vectors, model_name = embed_with_ollama(all_words)
        model_name = f"ollama: {model_name}"

    print(f"\n向量维度: {vectors.shape}")
    print(f"词数: {len(all_words)}")

    print("\n" + "=" * 60)
    print("1. 相似度分析")
    print("=" * 60)

    analyze_similarity_pairs(vectors, all_words, top_n=10)

    print("\n" + "=" * 60)
    print("2. 类别聚类分析")
    print("=" * 60)

    compare_within_category(vectors, all_words)

    print("\n" + "=" * 60)
    print("3. 可视化")
    print("=" * 60)

    print(f"\n图1: {model_name} - 词向量空间分布 (PCA降维)")
    visualize_embeddings(vectors, all_words, f"{model_name} - 词向量空间分布",
                         f"embedding_2d_{model_type}.png")

    print(f"\n图2: {model_name} - 余弦相似度热力图")
    sim_matrix = calculate_similarities(vectors)
    plot_similarity_heatmap(sim_matrix, all_words, f"{model_name} - Cosine Similarity",
                            f"similarity_heatmap_{model_type}.png")

    print(f"\n图3: {model_name} - 词汇距离网络 (相似度 > 0.5)")
    plot_word_distance_network(vectors, all_words, f"{model_name} - Word Distance Network",
                               threshold=0.5,
                               save_path=f"word_network_{model_type}.png")


if __name__ == "__main__":
    run_demo()