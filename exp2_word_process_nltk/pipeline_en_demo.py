#  NLP 处理流程完整演示（基于 Gensim 预训练词向量）

import sys
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api


# 首次运行需要下载 NLTK 的分词数据
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")


class Tee:
    """将 print 输出重定向到 log.txt（同时保留控制台输出）"""
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def clean_tokens(tokens, stop_words):
    """清洗分词结果：去标点 + 转小写 + 去停用词"""
    return [
        w.lower()
        for w in tokens
        if w.isalpha()
        and w.lower() not in stop_words
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
    """手动实现余弦相似度，结果范围 [-1, 1]，越接近1越相似"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    # 重定向输出
    log_file = open("log_en.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    # STEP 0：准备原始文本
    text = (
        "I love NLP, because it helps computers understand human language. "
        "Natural language processing is a fascinating field. "
        "Machine learning makes NLP more powerful every year."
    )

    print("=" * 60)
    print("【原始文本】")
    print(text)

    # STEP 1：分句（Sentence Tokenization）
    print("\n" + "=" * 60)
    print("【STEP 1】分句")

    sentences = sent_tokenize(text)
    for i, s in enumerate(sentences):
        print(f"  句子{i+1}：{s}")

    # STEP 2：分词（Word Tokenization）
    print("\n" + "=" * 60)
    print("【STEP 2】分词（以第1句为例）")

    words_raw = word_tokenize(sentences[0])
    print(f"  原始分词结果：{words_raw}")

    # STEP 3：清洗（去标点 + 转小写 + 去停用词）
    print("\n" + "=" * 60)
    print("【STEP 3】清洗文本")

    stop_words = set(stopwords.words("english"))
    words_clean = clean_tokens(words_raw, stop_words)
    print(f"  去标点 + 小写化 + 去停用词后：{words_clean}")

    # STEP 4：加载预训练词向量模型
    print("\n" + "=" * 60)
    print("【STEP 4】加载预训练词向量（首次运行会下载模型，约30MB）")

    model = api.load("glove-wiki-gigaword-50")
    print(f"  模型加载完成！词汇表大小：{len(model)} 个词，每个词向量维度：{model.vector_size}")

    # STEP 5：词嵌入（Word Embedding）—— 词 → 向量
    print("\n" + "=" * 60)
    print("【STEP 5】词嵌入：把词转换成向量")

    target_word = "language"
    vec = model[target_word]
    print(f"  '{target_word}' 的词向量（50维，只展示前10维）：")
    print(f"  {vec[:10].round(4)}  ...")
    print(f"  向量形状：{vec.shape}")

    # STEP 6：句子向量（把句子所有词向量取平均）
    print("\n" + "=" * 60)
    print("【STEP 6】句子向量（词向量取平均）")

    sentence_vecs = []
    for i, sent in enumerate(sentences):
        tokens = clean_tokens(word_tokenize(sent), stop_words)
        vec = sentence_to_vec(tokens, model)
        sentence_vecs.append(vec)
        print(f"  句子{i+1} tokens：{tokens}")
        print(f"  句子{i+1} 向量（前5维）：{vec[:5].round(4)} ...")
        print(f"  向量形状：{vec.shape}")

    # STEP 7：相似性检索（Similarity Search）
    print("\n" + "=" * 60)
    print("【STEP 7-A】词级别相似性：找与某个词最相近的词")

    similar_words = model.most_similar("language", topn=5)
    print(f"  与 'language' 最相似的5个词：")
    for word, score in similar_words:
        print(f"    {word:<20} 相似度={score:.4f}")

    print("\n【STEP 7-B】词对之间的相似度")
    pairs = [("nlp", "language"), ("love", "hate"), ("computer", "machine")]
    for w1, w2 in pairs:
        try:
            score = model.similarity(w1, w2)
            print(f"  similarity('{w1}', '{w2}') = {score:.4f}")
        except KeyError as e:
            print(f"  ⚠️  词汇表中不存在：{e}")

    print("\n【STEP 7-C】句子级别相似度（用余弦相似度）")

    print(f"  句子1：{sentences[0]}")
    print(f"  句子2：{sentences[1]}")
    print(f"  句子3：{sentences[2]}")
    print()

    s12 = cosine_similarity(sentence_vecs[0], sentence_vecs[1])
    s13 = cosine_similarity(sentence_vecs[0], sentence_vecs[2])
    s23 = cosine_similarity(sentence_vecs[1], sentence_vecs[2])

    print(f"  句子1 vs 句子2 相似度：{s12:.4f}")
    print(f"  句子1 vs 句子3 相似度：{s13:.4f}")
    print(f"  句子2 vs 句子3 相似度：{s23:.4f}")

    # STEP 8：词向量类比运算（经典演示）
    print("\n" + "=" * 60)
    print("【STEP 8】词向量类比运算：king - man + woman = ?")

    result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
    print("  计算 king - man + woman 最接近的词：")
    for word, score in result:
        print(f"    {word:<15} 相似度={score:.4f}")

    print("\n" + "=" * 60)
    print("✅ 流程演示完毕！")

    log_file.close()


if __name__ == "__main__":
    main()
