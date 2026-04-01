import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import jieba
from gensim.models import KeyedVectors

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
MODEL_PATH = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "stopwords_cn.txt")

# ========== 工具函数 ==========
def sent_tokenize_cn(text):
    """中文分句"""
    sentences = re.split(r'[。！？]', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_cn(text):
    """中文分词"""
    return list(jieba.cut(text))


def load_stopwords(filepath):
    """加载停用词表"""
    if not os.path.exists(filepath):
        print(f"  停用词文件不存在：{filepath}，将使用空停用词表")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def clean_tokens_cn(tokens, stopwords_cn):
    """清洗中文分词结果"""
    return [
        t for t in tokens
        if t.strip() and t not in stopwords_cn and not re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', t)
    ]


def sentence_to_vec(tokens, model):
    """将词向量取平均，得到句子向量"""
    vecs = [model[word] for word in tokens if word in model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def reduce_to_2d(vectors):
    """PCA降维到2D"""
    vectors = np.array(vectors)
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors)


# ========== 可视化函数 ==========
def plot_sentences(sentence_vecs, sentences, ax=None):
    """
    绘制句子散点图
    每个点代表一个句子，标注句子序号和简短文本
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 降维
    coords = reduce_to_2d(sentence_vecs)

    # 绘制散点
    ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=150, alpha=0.7, edgecolors='white')

    # 标注文本
    for i, (x, y) in enumerate(coords):
        # 截取前10个字符作为标签
        label = f"{i+1}. {sentences[i][:10]}..."
        ax.annotate(label, (x, y), fontsize=10, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points')

    ax.set_title('句子向量分布图', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    return ax


def plot_words(word_vecs, words, title='词向量分布图', ax=None):
    """
    绘制词的散点图
    每个点代表一个词
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 降维
    coords = reduce_to_2d(word_vecs)

    # 绘制散点
    ax.scatter(coords[:, 0], coords[:, 1], c='coral', s=100, alpha=0.7, edgecolors='white')

    # 标注词
    for i, (x, y) in enumerate(coords):
        ax.annotate(words[i], (x, y), fontsize=11, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    return ax


def plot_word_clusters(word_vecs, words, ax=None):
    """
    绘制词汇聚类图
    在同一坐标系中展示所有词，观察语义聚类
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # 降维
    coords = reduce_to_2d(word_vecs)

    # 绘制散点，使用不同颜色区分
    colors = plt.cm.Set3(np.linspace(0, 1, len(words)))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=120, alpha=0.8, edgecolors='gray')

    # 标注词
    for i, (x, y) in enumerate(coords):
        ax.annotate(words[i], (x, y), fontsize=11, ha='center',
                   xytext=(0, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_title('词汇聚类图 - 语义相近的词应聚集在一起', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    return ax


# ========== 主流程 ==========
def main():
    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在：{MODEL_PATH}")
        print("请下载词向量模型，参见 pipeline_cn_demo.py 文件头部注释")
        return

    print("=" * 60)
    print("加载词向量模型...")
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    print(f"模型加载完成！词汇表大小：{len(model)}，向量维度：{model.vector_size}")

    # 加载停用词
    stopwords_cn = load_stopwords(STOPWORDS_PATH)

    # 示例文本
    text = (
        "我喜欢自然语言处理，因为它能帮助计算机理解人类语言。"
        "自然语言处理是一个迷人的领域。"
        "机器学习让自然语言处理每年都变得更强大。"
        "深度学习在图像识别方面取得了巨大成功。"
        "人工智能正在改变我们的生活方式。"
    )

    print("\n" + "=" * 60)
    print(f"原始文本：{text}")

    # 分句
    sentences = sent_tokenize_cn(text)
    print(f"\n分句结果（{len(sentences)}句）：")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

    # 分词 + 清洗
    all_words = []
    sentence_tokens = []
    for sent in sentences:
        tokens = clean_tokens_cn(tokenize_cn(sent), stopwords_cn)
        sentence_tokens.append(tokens)
        all_words.extend(tokens)

    # 去重保留顺序
    unique_words = list(dict.fromkeys(all_words))
    print(f"\n所有词汇：{unique_words}")

    # 获取句子向量
    sentence_vecs = []
    valid_sentences = []
    for i, tokens in enumerate(sentence_tokens):
        vec = sentence_to_vec(tokens, model)
        if vec is not None:
            sentence_vecs.append(vec)
            valid_sentences.append(sentences[i])

    # 获取词向量
    word_vecs = []
    valid_words = []
    for word in unique_words:
        if word in model:
            word_vecs.append(model[word])
            valid_words.append(word)

    print(f"\n有效句子数：{len(sentence_vecs)}")
    print(f"有效词汇数：{len(word_vecs)}")

    if len(sentence_vecs) < 2 or len(word_vecs) < 2:
        print("数据不足，无法可视化")
        return

    # ========== 绘制三张图 ==========
    print("\n" + "=" * 60)
    print("生成可视化图表...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图1：句子散点图
    plot_sentences(np.array(sentence_vecs), valid_sentences, ax=axes[0])

    # 图2：词的散点图
    plot_words(np.array(word_vecs), valid_words, title='词向量分布图', ax=axes[1])

    # 图3：词汇聚类图
    plot_word_clusters(np.array(word_vecs), valid_words, ax=axes[2])

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(os.path.dirname(__file__), 'pipeline_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至：{output_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("可视化完成！")


if __name__ == "__main__":
    main()
