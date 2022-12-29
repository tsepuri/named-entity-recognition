# parsed and tagged CONLL data or Twitter data required
from src.LDA.lda import LDA
from src.CoNLLExtractor.preprocess import default_reading
from src.LDA.ModelBuilders.adversial_training import AdversialTraining
#import nltk
import pickle
class OfflineTraining():
    def __init__(self, data_type, window_size=1, noise=False, masking=False):
        self.data_type = data_type
        self.data = {}
        self.context_probs = {}
        self.entity_probs = {}
        self.window_size = window_size
        self.classes = []
        self.noise = noise
        self.masking = masking
        self.context_map = {}
        self.class_probs = {}
        self.new_context_map = {}
        self.test_data = {
            'named_entities': ["Star Wars", "Game of Thrones", "Harry Potter", "Lord of the Rings", "The Matrix", "The Avengers", "Titanic", "Jurassic Park", "Star Trek", "Doctor Who"],
            'queries' : ["I can't wait to see the new Star Wars movie!", "Star Wars is life", "Just played the best game of my life!", "Reading a great book on natural language processing.", "Listening to my favorite music on Spotify.", "I love Game of Thrones!", "Harry Potter is my all-time favorite book series.", "The Lord of the Rings movies are amazing.", "I love the Matrix trilogy.", "The Avengers are my favorite Marvel superheroes.", "Titanic is one of my favorite movies.", "Jurassic Park is a classic!", "I'm a big fan of Star Trek.", "Doctor Who is the best TV show ever!"],
            'actual_classes': ["MOV", "MOV", "PLA", "BOK", "NON", "TV", "BOK", "MOV", "MOV", "MOV", "MOV", "MOV", "TV", "TV"],
            'classes': [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        }
        self.all_contexts = set()
        self.entities = set()
        self.entity_count = 0
    def summarize(self):
        data = {}
        actual_entities = set()
        if self.data_type == "test_data":
            data = self.test_data
            data = self.test_data
        elif self.data_type == "conll_data":
            data = default_reading()
            self.data = data
            self.classes = data["classes"]
        self.context_map = {}
        entityAdded = False
        at = AdversialTraining()
        if self.masking:  
            data['queries'] = at.mask_entities(data['queries'], data['named_entities'])
            self.data['queries'] = data['queries']
        for index, word in enumerate(data['queries']):
            if index % 10000 == 0:
                print(f"Word {index} of {len(data['queries'])}")
            actual_entities.add(data['named_entities'][index])
            # Removing the first condition allows training on all contexts
            if data['named_entities'][index] != "O" and not entityAdded:
                if self.noise:
                    # Want to set to a different entity but not to outside label
                    classes_to_pass = data['classes'].copy()
                    classes_to_pass.remove("O")
                    data['named_entities'][index] = at.adversial_noise(index, data['queries'], data['named_entities'], data['named_entities'][index], classes_to_pass)
                for window_size in range(0, self.window_size*3+1):
                    num_sen = word[1]
                    curr_word = word[0]
                    self.entity_count = self.entity_count + 1
                    new_context = []
                    leftWindow = 0
                    rightWindow = 0
                    currentLeft = index - 1
                    currentRight = index + 1
                    full_entity = curr_word
                    full_entity_tags = [data['named_entities'][index]]
                    if window_size == 0:
                        continue
                    # While within window and in the same sentence
                    if window_size % 3 != 0 or (window_size == 3 and self.window_size >= 3):
                        while leftWindow < window_size and currentLeft >= 0 and data['queries'][currentLeft][1] == num_sen:
                            if data['named_entities'][currentLeft] != "O":
                                full_entity = data['queries'][currentLeft][0] + " " + full_entity
                                full_entity_tags = [data['named_entities'][currentLeft]] + full_entity_tags
                            else:
                                new_context = [data['queries'][currentLeft][0]] + new_context
                                leftWindow = leftWindow + 1
                            currentLeft = currentLeft - 1
                    # Hiding entity
                    new_context.append("#")
                    if window_size % 2 != 0 or (window_size == 2 and self.window_size >= 2):
                        while rightWindow < window_size and currentRight < len(data['queries']) and data['queries'][currentRight][1] == num_sen:
                            if data['named_entities'][currentRight] != "O":
                                full_entity = full_entity + " " + data['queries'][currentRight][0]
                                full_entity_tags.append(data['named_entities'][currentRight])
                            else:
                                new_context.append(data['queries'][currentRight][0])
                                rightWindow = rightWindow + 1
                            currentRight = currentRight + 1
                    new_context = tuple(new_context)
                    self.all_contexts.add(new_context)
                    entityAdded = True
                    if full_entity not in self.context_map:
                        self.context_map[full_entity] = {}
                    if "total_num" not in self.context_map[full_entity]:
                        self.context_map[full_entity]["total_num"] = 1
                    else: 
                        self.context_map[full_entity]["total_num"] = self.context_map[full_entity]["total_num"] + 1
                    if new_context not in self.context_map[full_entity]:
                        self.context_map[full_entity][new_context] = (0,full_entity_tags)
                    self.context_map[full_entity][new_context] = (self.context_map[full_entity][new_context][0] + 1, self.context_map[full_entity][new_context][1])
            else:
                entityAdded = False
        print(actual_entities)
        return self.context_map
        # WS-LDA
        # Store learned prbabiltiies into class index Ic, named entity index is Ie
    def train(self):
        self.summarize()
        lda = LDA(self.classes, self.context_map)
        lda.preprocess()
        lda.train()
        self.context_probs = {}
        for context in self.all_contexts:
            self.context_probs[context] = {}
            for curr_class in self.classes:
                self.context_probs[context][curr_class] = lda.infer(context, curr_class)
        #self.all_entities()
        self.entity_prob()
        print("Trained!")
        return lda
    
    def all_entities(self):
        entity_count = 0
        for index, word in enumerate(self.data["queries"]):
            word = word[0]
            for context in self.all_contexts:
                if self._query_contains_context(self.data["queries"], index, context):
                    entity = self._query_remainder_entity(self.data["queries"], index, context)
                    self.entities.add(entity)
                    entity_count = entity_count + 1
                    if entity not in self.new_context_map:
                        self.new_context_map[entity] = {}
                    if "total_num" not in self.new_context_map[entity]:
                        self.new_context_map[entity]["total_num"] = 1
                    else: 
                        self.new_context_map[entity]["total_num"] = self.new_context_map[entity]["total_num"] + 1
                    if context not in self.new_context_map[entity]:
                        self.new_context_map[entity][context] = (0, [])
                    self.new_context_map[entity][context] = (self.new_context_map[entity][context][0] + 1, self.new_context_map[entity][context][1])
        print(f"Total expected entities: {entity_count}")
        self.entity_count = max(entity_count, self.entity_count)
        return self._high_quality_entities()
        
    def entity_prob(self):
        for each_class in self.classes:
            self.class_probs[each_class] = {}
        for entity, val in self.context_map.items():
            curr_class_map = {}
            total_labels = 0
            self.entity_probs[entity] = self.context_map[entity]["total_num"] / self.entity_count
            for context, info in val.items():
                if "total_num" in context:
                    continue
                else:
                    next_tuple = info
                    count = next_tuple[0]
                    labels = next_tuple[1]
                    for label in labels:
                        total_labels = total_labels + 1
                        if label not in curr_class_map:
                            curr_class_map[label] = 0
                        curr_class_map[label] = curr_class_map[label] + 1
            for each_class in self.classes:
                self.class_probs[each_class][entity] = curr_class_map[label]/total_labels
                        

    
    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)
        with open(filename, 'rb') as handle:
            saved = pickle.load(handle)
        return self == saved
    
    # Online prediction algorithm
    def predict(self, words, type="direct"):
        for index, word in enumerate(words):
            done = False
            for i in range(index, index+10):
                if done:
                    break
                for j in range(i, index+10):
                    if done: 
                        break
                    if words[j][1] != words[i][1]:
                        done = True
                    t = tuple(words[:i] + ["#"] + words[j+1:])
                    #model.in
                        

        
    def online_prediction(self, queries, model, K=5):
        R = []
        for i in range(len(queries)):
            for j in range(i, len(queries)):
                # extract named entity and context
                e = " ".join(queries[i:j+1])
                t = tuple(queries[:i] + ["#"] + queries[j+1:])
                if e in self.entity_probs or t in self.context_probs:#True:
                    # iterate over the classes
                    for c in self.classes:
                        # create a new recognition result
                        r = {"triple": (e, t, c), "prob": 0}

                        # compute the probabilities using the indexes
                        if e in self.entity_probs:
                            Pr_e = self.entity_probs[e]
                            Pr_c_given_e = self.class_probs[c][e]
                        else:
                            Pr_e = 1/self.entity_count
                            Pr_c_given_e = 1/len(self.class_probs.keys())
                        if t in self.context_probs:
                            Pr_t_given_c = self.context_probs[t][c]
                        else:
                            Pr_t_given_c = -1
                            for word in t:
                                if word in self.context_probs and word != "#":
                                    Pr_t_given_c = self.context_probs[tuple(word,)][c]
                                    break
                            if Pr_t_given_c == -1:
                                continue
                        # compute overall probability for the recognition result
                        r["prob"] = Pr_e * Pr_c_given_e * Pr_t_given_c
                        R.append(r)
                else:
                    for c in self.classes:
                        r = {"triple": (e, t, c), "prob": 0}
                        Pr_e = 1/self.entity_count
                        Pr_c_given_e = 1/len(self.class_probs.keys())
                        r["prob"] = model.infer(t, c) * Pr_e * Pr_c_given_e
                        R.append(r)
        # srt results by probability
        R.sort(key=lambda x: x["prob"], reverse=True)
        # truncate results to top K
        if len(R) == 0:
            return []
        print(R)
        R = R[:K]
        return R


    def _high_quality_entities(self):
        # nltk could be used here
        # could also check the number of contexts present in the entity
        return self.entities


    def _query_remainder_entity(self, queries, index, context):
        context_index = 0
        began = False
        entity = ""
        entity_max = 4
        while context_index < len(context) and context_index < self.window_size*2+1:
            if context[context_index] == "#":
                began = True
                entity = entity + " " + queries[index][0]
            if began:
                if str.lower(context[context_index]).strip() == str.lower(queries[index][0]).strip():
                    break
                else:
                    entity = entity + " " + queries[index][0]
            else:
                context_index = context_index + 1
            if index+1 < len(queries) and queries[index+1][1] == queries[index][1]:
                index = index + 1
            else:
                break
        return entity

    def _query_contains_context(self, queries, index, context):
        words = context
        max_misses = 0
        misses = 0
        context_index = 0
        hash_reached = True
        entity_max = 3
        entity_len = 1
        while context_index < len(words) and context_index < self.window_size*2+1:
            if words[context_index] != "#":
                current_contained = str.lower(words[context_index]).strip() == str.lower(queries[index][0]).strip()
                if not current_contained:
                    # If we are in the right context, we do not know how long the entity is
                    if hash_reached:
                        if index+1 < len(queries) and queries[index+1][1] == queries[index][1]:
                            if entity_len+1 > entity_max:
                                misses = misses + 1
                            else:
                                index = index + 1
                                entity_len = entity_len + 1
                                continue
                        else:
                            misses = misses + 1
                    else:
                        misses = misses + 1
            else:
                hash_reached = True
            context_index = context_index + 1
            if index+1 < len(queries) and queries[index+1][1] == queries[index][1]:
                index = index + 1
            else:
                break
        return misses <= max_misses




#ot = OfflineTraining("conll_data")
#ot.summarize()
#ot.train()
#rint(ot.save("small.dat"))
#print(ot.all_entities())
