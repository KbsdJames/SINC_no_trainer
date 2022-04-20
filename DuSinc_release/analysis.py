import pdb
training_set = []
with open("E:/Python_relevant/Paddle/DuSinc_release/test_query_1.txt", "r", encoding="utf-8") as f:
    for userline in f:
        userline = eval(userline)
        training_set.append(userline)

key_len = []

for sample in training_set:
    for conversation in sample['conversation']:
        if(conversation['role'] == 'bot'):
            if conversation['use_kg_label'] == 'true':
                assert(len(conversation.keys()) == 6)
            else:
                assert(len(conversation.keys()) == 4)

print(key_len)
pdb.set_trace()
print(training_set[0].keys())
