import os
import json


# make a list of all the folders in the folder ../PragmatiCQA-sources
def list_folders(directory="../PragmatiCQA-sources"):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
folders = list_folders()

topics = set()

with open("../PragmatiCQA/data/test.jsonl", 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'topic' in data:
            topics.add(data['topic'])

# with open("../PragmatiCQA/data/val.jsonl", 'r') as file:
#     for line in file:
#         data = json.loads(line)
#         if 'topic' in data:
#             topics.add(data['topic'])


# print("Folders in ../PragmatiCQA-sources:")
# print(folders)

# print("\nTopics in ../PragmatiCQA/data/test.jsonl:")
# print(topics)

for topic in topics:
    if topic not in folders:
        print(topic)

# Snoopy
# Spirited Away
# New York Yankees
# Po