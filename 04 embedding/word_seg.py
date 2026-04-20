# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import os
from pathlib import Path

import jieba
from utilsSpecial import files_processing

# 相对「本脚本」定位目录，避免依赖终端当前工作目录（否则从项目根运行会找不到 ./journey_to_the_west）
BASE_DIR = Path(__file__).resolve().parent
source_folder = BASE_DIR / "journey_to_the_west"
segment_folder = BASE_DIR / "journey_to_the_west"


# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list, segment_out_dir, stopwords=None):
    if stopwords is None:
        stopwords = []
    segment_out_dir = Path(segment_out_dir)
    segment_out_dir.mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(file_list):
        segment_out_name = segment_out_dir / "segment_{}.txt".format(i)
        with open(file, "rb") as f:
            document = f.read()
        # jieba 需要 str；原先用 bytes 可能按字节切分，中文应整体按字符
        text = document.decode("utf-8", errors="replace")
        document_cut = jieba.cut(text)
        sentence_segment = []
        for word in document_cut:
            if word not in stopwords:
                sentence_segment.append(word)
        result = " ".join(sentence_segment)
        result = result.encode("utf-8")
        with open(segment_out_name, "wb") as f2:
            f2.write(result)


if __name__ == "__main__":
    # 对 source 中的 txt 分词，输出到 segment_folder（排除已生成的 segment_*.txt，避免重复处理）
    raw_list = files_processing.get_files_list(source_folder, postfix="*.txt")
    file_list = [
        p for p in raw_list if not Path(p).name.startswith("segment_")
    ]
    if not file_list:
        print(
            "未找到任何 .txt 文件，已跳过。请确认目录存在且含 txt：",
            source_folder,
        )
    else:
        print("将处理 {} 个文件，输出目录：{}".format(len(file_list), segment_folder))
        segment_lines(file_list, segment_folder)
