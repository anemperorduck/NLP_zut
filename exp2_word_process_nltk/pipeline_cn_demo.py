#  中文 NLP 处理流程完整演示（基于 jieba 分词 + 预训练词向量）
#
#  词向量模型下载（选一个）：
#  1. Tencent AI Lab 词向量: https://ai.tencent.com/ailab/nlp/embedding.html (~6GB)
#  2. Chinese-Word-Vectors: https://github.com/Embedding/Chinese-Word-Vectors (~1.5GB)
#     推荐下载: sgns.target.word-word.dynwin5.thr10.neg5.dim300
#
#  下载后将模型文件放到本目录下，或修改 MODEL_PATH 变量

import sys
import re
import os
import numpy as np
import jieba
from gensim.models import KeyedVectors


# ========== 配置 ==========
MODEL_PATH = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"  # 修改为你的模型路径
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "stopwords_cn.txt")


# ========== 工具类 ==========
class Tee:
    """将 print 输出重定向到 log_cn.txt（同时保留控制台输出）"""
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ========== 核心函数 ==========
def sent_tokenize_cn(text):
    """中文分句：按句号、感叹号、问号分割"""
    sentences = re.split(r'[。！？]', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_cn(text):
    """中文分词：使用 jieba"""
    return list(jieba.cut(text))


def load_stopwords(filepath):
    """加载停用词表"""
    if not os.path.exists(filepath):
        print(f"  ⚠️ 停用词文件不存在：{filepath}，将使用空停用词表")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def clean_tokens_cn(tokens, stopwords_cn):
    """清洗中文分词结果：过滤停用词、标点、空白"""
    return [
        t for t in tokens
        if t.strip() and t not in stopwords_cn and not re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', t)
    ]


def sentence_to_vec(tokens, model):
    """将词向量取平均，得到句子级别的向量"""
    vecs = []
    skipped = []
    for word in tokens:
        if word in model:
            vecs.append(model[word])
        else:
            skipped.append(word)
    if skipped:
        print(f"  ⚠️  词汇表中找不到，已跳过：{skipped}")
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cosine_similarity(v1, v2):
    """余弦相似度，结果范围 [-1, 1]，越接近1越相似"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    # 重定向输出
    log_file = open("log_cn.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    # STEP 0：准备原始文本
    text = (
        "我喜欢自然语言处理，因为它能帮助计算机理解人类语言。"
        "自然语言处理是一个迷人的领域。"
        "机器学习让自然语言处理每年都变得更强大。"
    )

    print("=" * 60)
    print("【原始文本】")
    print(text)

    # STEP 1：分句
    print("\n" + "=" * 60)
    print("【STEP 1】分句")

    sentences = sent_tokenize_cn(text)
    for i, s in enumerate(sentences):
        print(f"  句子{i+1}：{s}")

    # STEP 2：分词
    print("\n" + "=" * 60)
    print("【STEP 2】分词（以第1句为例）")

    words_raw = tokenize_cn(sentences[0])
    print(f"  原始分词结果：{words_raw}")

    # STEP 3：清洗（去停用词 + 去标点）
    print("\n" + "=" * 60)
    print("【STEP 3】清洗文本")

    stopwords_cn = load_stopwords(STOPWORDS_PATH)
    print(f"  停用词表大小：{len(stopwords_cn)} 个词")

    words_clean = clean_tokens_cn(words_raw, stopwords_cn)
    print(f"  去停用词 + 去标点后：{words_clean}")

    # STEP 4：加载预训练词向量模型
    print("\n" + "=" * 60)
    print("【STEP 4】加载预训练词向量")

    if not os.path.exists(MODEL_PATH):
        print(f"  ⚠️ 模型文件不存在：{MODEL_PATH}")
        print("  请下载词向量模型，参见文件头部注释")
        print("  程序退出。")
        log_file.close()
        return

    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    print(f"  模型加载完成！词汇表大小：{len(model)} 个词，向量维度：{model.vector_size}")

    # STEP 5：词嵌入
    print("\n" + "=" * 60)
    print("【STEP 5】词嵌入：把词转换成向量")

    target_word = "语言"
    if target_word in model:
        vec = model[target_word]
        print(f"  '{target_word}' 的词向量（前10维）：")
        print(f"  {vec[:10].round(4)}  ...")
        print(f"  向量形状：{vec.shape}")
    else:
        print(f"  ⚠️ '{target_word}' 不在词汇表中")

    # STEP 6：句子向量
    print("\n" + "=" * 60)
    print("【STEP 6】句子向量（词向量取平均）")

    sentence_vecs = []
    for i, sent in enumerate(sentences):
        tokens = clean_tokens_cn(tokenize_cn(sent), stopwords_cn)
        vec = sentence_to_vec(tokens, model)
        sentence_vecs.append(vec)
        print(f"  句子{i+1} tokens：{tokens}")
        if vec is not None:
            print(f"  句子{i+1} 向量（前5维）：{vec[:5].round(4)} ...")
        else:
            print(f"  句子{i+1} 向量：无法生成（无有效词）")

    # STEP 7：相似性检索
    print("\n" + "=" * 60)
    print("【STEP 7-A】词级别相似性：找与某个词最相近的词")

    target = "语言"
    if target in model:
        similar_words = model.most_similar(target, topn=5)
        print(f"  与 '{target}' 最相似的5个词：")
        for word, score in similar_words:
            print(f"    {word:<20} 相似度={score:.4f}")
    else:
        print(f"  ⚠️ '{target}' 不在词汇表中")

    print("\n【STEP 7-B】词对之间的相似度")
    pairs = [("语言", "自然"), ("喜欢", "爱"), ("计算机", "机器")]
    for w1, w2 in pairs:
        try:
            score = model.similarity(w1, w2)
            print(f"  similarity('{w1}', '{w2}') = {score:.4f}")
        except KeyError as e:
            print(f"  ⚠️  词汇表中不存在：{e}")

    print("\n【STEP 7-C】句子级别相似度")

    print(f"  句子1：{sentences[0]}")
    print(f"  句子2：{sentences[1]}")
    print(f"  句子3：{sentences[2]}")
    print()

    if all(v is not None for v in sentence_vecs):
        s12 = cosine_similarity(sentence_vecs[0], sentence_vecs[1])
        s13 = cosine_similarity(sentence_vecs[0], sentence_vecs[2])
        s23 = cosine_similarity(sentence_vecs[1], sentence_vecs[2])

        print(f"  句子1 vs 句子2 相似度：{s12:.4f}")
        print(f"  句子1 vs 句子3 相似度：{s13:.4f}")
        print(f"  句子2 vs 句子3 相似度：{s23:.4f}")
    else:
        print("  ⚠️ 部分句子向量无效，无法计算相似度")

    # STEP 8：词向量类比运算
    print("\n" + "=" * 60)
    print("【STEP 8】词向量类比运算：国王 - 男人 + 女人 = ?")

    try:
        result = model.most_similar(positive=["国王", "女人"], negative=["男人"], topn=3)
        print("  计算 国王 - 男人 + 女人 最接近的词：")
        for word, score in result:
            print(f"    {word:<15} 相似度={score:.4f}")
    except KeyError as e:
        print(f"  ⚠️  词汇表中不存在：{e}")

    print("\n" + "=" * 60)
    print("✅ 流程演示完毕！")

    log_file.close()


if __name__ == "__main__":
    main()
