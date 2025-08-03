import os
import json
import hashlib

def get_cache_key(topic, question):
        key_str = f"{topic}|{question}"
        return hashlib.md5(key_str.encode()).hexdigest()


# make a list of all the folders in the folder ../PragmatiCQA-sources
# def list_folders(directory="../PragmatiCQA-sources"):
#     return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
# folders = list_folders()



s = set()

with open("../PragmatiCQA/data/val.jsonl", 'r') as file:
    for line in file:
        data = json.loads(line)
        topic = data.get('topic')
        if topic in ["Snoopy"]:
            continue
        qas = data.get('qas')[0]
        question = qas['q']
        hash = get_cache_key(topic, question)
        if hash in s:
             print(f"Duplicate found: {hash} for topic: {topic} and question: {question}")
        else:
            s.add(hash)

# with open("../PragmatiCQA/data/val.jsonl", 'r') as file:
#     for line in file:
#         data = json.loads(line)
#         if 'topic' in data:
#             topics.append(data['topic'])


# print("Folders in ../PragmatiCQA-sources:")
# print(folders)

# print("\nTopics in ../PragmatiCQA/data/test.jsonl:")
# print(topics)
# count = 0
# for topic in topics:
#     if topic not in folders:
#         count += 1

# print(count)

# A Nightmare on Elm Street (2010 film)
# Popeye
# The Wonderful Wizard of Oz (book)
# Alexander Hamilton