import sys
import json
import pdb


def conv_to_gen_response(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为知识对话生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"].replace(" ", "")
            if is_test:
                context = [uttr["utterance"].replace(" ", "")
                           for uttr in data["conversation"][:-1]]
                if "use_knowledge" in data["conversation"][-1]:
                    knowledge = data["conversation"][-1]["use_knowledge"].replace(
                        " ", "")
                else:
                    knowledge = ""
                # 对部分过长的知识进行截断，只保留前256个字符
                knowledge = knowledge.replace(
                    "\n", " ").replace("\t", " ")[:256]
                outstr = knowledge + "[SEP]" + location + \
                    "[SEP]" + "[SEP]".join(context)
                fout.write(outstr.rstrip().replace("\n", " ") + "\n")
                continue
            for uttr in data["conversation"]:
                if is_test:
                    context.append(uttr["utterance"].replace(" ", ""))
                    continue
                if "use_kg_label" in uttr:
                    if uttr["use_kg_label"] == "true":
                        try:
                            knowledge = uttr["use_knowledge"].replace(" ", "").replace(
                                "\n", " ").replace("\t", " ")
                        except:
                            print(json.dumps(uttr, ensure_ascii=False, indent=2))
                    else:
                        knowledge = ""
                    knowledge = knowledge.replace("\n", " ").replace("\t", " ")[:256]
                    response = uttr["utterance"].replace(" ", "")
                    outstr = knowledge + "[SEP]" + location + \
                        "[SEP]" + "[SEP]".join(context) + "\t" + response
                    fout.write(outstr.rstrip().replace("\n", " ") + "\n")
                context.append(uttr["utterance"].replace(
                    " ", "").replace("\n", " "))
    fout.close()



conv_to_gen_response("DuSinc_release/train.txt",
                  "DuSinc/response/train.csv")
conv_to_gen_response("DuSinc_release/dev.txt", "DuSinc/response/dev.csv")
conv_to_gen_response("DuSinc_release/test_dial_1.txt",
                  "DuSinc/response/test.csv", is_test=True)


