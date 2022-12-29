import numpy as np
import tomotopy as tp
import random
# Word - _iw_ - Context
# Named entity - Document - _bw_
# Class - Topic - _zn_
class LDA():
    def __init__(self, classes, context_map, gamma=0.8):
        self.classes = classes
        self.context_map = context_map
        self.gamma = gamma
        # partially labeled Latent Dirichlet allocation
        self.model = tp.PLDAModel()
    
    def preprocess(self):
        localTotalCount = 0
        for entity, val in self.context_map.items():
            prior_prob = np.zeros(len(self.classes))
            for context, info in val.items():
                if "total_num" in context:
                    localTotalCount = info
                else:
                    next_tuple = info
                    count = next_tuple[0]
                    labels = next_tuple[1]
                    #strings = context.split("#")
                    #next_doc = [strings[0]] + [entity] + [strings[1]]
                    next_doc = list(context)
                    if random.random() < self.gamma:
                        print(labels)
                        self.model.add_doc(next_doc, labels=labels)
                    else:
                        self.model.add_doc(next_doc)
                    #prior_prob[list(self.classes).index(labels)] = prior_prob[self.classes.index(labels)] + count
            #self.model.set_word_prior(entity, [np.divide(prob, localTotalCount) for prob in prior_prob])
    
    def train(self):
        for i in range(0, 1000, 10):
            self.model.train(10)
            print(f"Iteration {i} and log-likelihood: {self.model.ll_per_word}")
        print(self.model.summary())

    def new_document(self, words):
        return self.model.make_doc(words)
    
    def infer(self, words, label):
        index = self.label_index(label)
        if index != -1:
            return self.model.infer(self.new_document(words))[0][index]
        return -1

    def label_index(self, label):
        topic_dicts = self.model.topic_label_dict
        for k in range(self.model.k):
            if topic_dicts[k] == label:
                return k
        return -1
