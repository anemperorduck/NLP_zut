import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import string

# 首次运行必须下载数据包
# nltk.download('punkt')           # 分词模型
# nltk.download('stopwords')       # 停用词表
# nltk.download('averaged_perceptron_tagger')  # 词性标注器
# nltk.download('averaged_perceptron_tagger_eng')  # 新版NLTK需要额外下载英语包
# nltk.download('punkt_tab')


def text_preprocessing_pipeline(text):
    """
    标准文本预处理流程：分句→分词→去停用词/标点→词干提取→词性标注
    """
    # 1. 分句（适合长文档处理）
    sentences = sent_tokenize(text)
    print(f"分句结果: {sentences}\n")
    
    processed_sentences = []
    
    for sent in sentences:
        # 2. 分词（基于Penn Treebank规则）
        tokens = word_tokenize(sent)
        
        # 3. 去除停用词和标点
        stop_words = set(stopwords.words('english'))
        tokens = [w.lower() for w in tokens 
                  if w.lower() not in stop_words and
                    w not in string.punctuation]
        
        # 4. 词干提取（将running/run/ran→run）
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(w) for w in tokens]
        
        # 5. 词性标注（为后续NER或句法分析准备）
        pos_tags = pos_tag(stemmed)
        
        processed_sentences.append(pos_tags)
    
    return processed_sentences

sample_text = "NLTK is a powerful library! It provides tools for NLP tasks including tokenization and stemming."
result = text_preprocessing_pipeline(sample_text)
print("最终处理结果：")
for sent in result:
    print(sent)


