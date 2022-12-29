from dataclasses import dataclass
from dotenv import load_dotenv
import os
DOC_START = "-DOCSTART- -X- -X- O\n"

classes = set()
load_dotenv()
@dataclass
class Input():
    word: str
    pos: str
    pos_loc: str
    ner: str

def read_from_file(filename=load_dotenv("")):
    named_entities = []
    queries = []
    with open(filename, encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
    queries = []
    sentence = 0
    for line in lines:
        if line == DOC_START:
            continue
        if line == "\n":
            sentence = sentence + 1
            continue
        words = input_format(line)
        if words:
            queries.append(tuple([words.word, sentence]))
            named_entities.append(words.ner)
            classes.add(words.ner)
    return {
        'queries': queries,
        'named_entities': named_entities,
        'classes': classes
    }

def input_format(line):
    words = line.split()
    if words:
        return Input(word=words[0], pos=words[1], pos_loc=words[2], ner=words[3])
    return []

def default_reading(type="train"):
    
    #os.getenv("CONLL_DATASET_FOLDER")
    return read_from_file(os.path.join("/Users/tspri/Downloads/conll2003/", f"{type}.txt"))

print(default_reading())