import sys
import json
import pdb


def conv_to_gen_query(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = "".join(data["user_topical"])
            topical = topical.replace(" ", "")
            location = data["user_location"].replace(" ", "")
            for uttr in data["conversation"]:
                utterence = uttr["utterance"]
                utterence = utterence.replace(" ", "")
                if is_test:
                    context.append(utterence)
                    continue
                if uttr["role"] == "bot":
                    if "use_query" in uttr:
                        query = uttr["use_query"].replace(" ", "")
                    else:
                        query = "不检索"
                    outstr = location + "[SEP]" + \
                        "[SEP]".join(context) + '\t' + query
                    fout.write(outstr.strip().replace("\n", " ") + "\n")
                context.append(utterence)
            if is_test:
                outstr = location + \
                    "[SEP]" + "[SEP]".join(context)
                fout.write(outstr.strip().replace("\n", " ") + "\n")
    fout.close()


conv_to_gen_query("DuSinc_release/train.txt",
                  "DuSinc/dialogue/train.csv")
conv_to_gen_query("DuSinc_release/dev.txt", "DuSinc/dialogue/dev.csv")
conv_to_gen_query("DuSinc_release/test_query_1.txt",
                  "DuSinc/dialogue/test.csv", is_test=True)

