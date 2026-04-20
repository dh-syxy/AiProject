# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from pathlib import Path
from gensim.models import word2vec
import multiprocessing

BASE_DIR = Path(__file__).resolve().parent
segment_folder = BASE_DIR / "journey_to_the_west"
if not segment_folder.is_dir():
    raise FileNotFoundError(
        "找不到分词结果目录：{}（请先在同目录运行 word_seg.py）".format(segment_folder)
    )


class SegmentGlobSentences:
    """只读取目录下 segment_*.txt；每行按空白切分为词列表。每次 __iter__ 重新读文件，供 Word2Vec 多轮训练。"""

    def __init__(self, directory: Path, glob_pattern: str = "segment_*.txt"):
        self.directory = Path(directory)
        self.glob_pattern = glob_pattern

    def __iter__(self):
        paths = sorted(self.directory.glob(self.glob_pattern))
        if not paths:
            raise ValueError(
                "目录 {} 下没有匹配 {} 的文件".format(
                    self.directory, self.glob_pattern
                )
            )
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line.split()


# 切分之后的语料：仅 segment_*.txt（不含 three_kingdoms.txt 等其它 txt）
sentences = SegmentGlobSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, vector_size=160, window=3, min_count=1)
print(model.wv['曹操'])
print(model.wv.most_similar('曹操', topn=5))
print(model.wv.similarity('曹操', '诸葛亮'))
print(model.wv.similarity('曹操', '曹孟德'))
print(model.wv.most_similar(positive=['曹操', '诸葛亮'], negative=['张飞']))
# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())
# 保存模型
_models_dir = BASE_DIR / "models"
_models_dir.mkdir(parents=True, exist_ok=True)
model2.save(str(_models_dir / "word2Vec.model"))
print(model2.wv.similarity('曹操', '诸葛亮'))
print(model2.wv.similarity('曹操', '魏王'))
print(model2.wv.most_similar(positive=['曹操', '魏王'], negative=['魏王']))