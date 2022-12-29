import random
class AdversialTraining():
    def __init__(self):
        self.threshold = 0.8
    # mask all entities followed by at least three non-entity words
    def mask_entities(self, queries, entities):
        current_streak = 0
        last_seen_index = -1
        last_entity_size = 0
        for index, word in enumerate(queries):
            if entities[index] == 'O':
                current_streak = current_streak + 1
                if current_streak == 3:
                    if last_seen_index != -1:
                        for entity_index in range(last_seen_index+1-last_entity_size, last_seen_index+1):
                            queries[entity_index] = '[MASK]'
                        last_seen_index = -1
            else:
                if current_streak != 0:
                    last_entity_size = 0
                current_streak = 0
                last_entity_size = last_entity_size + 1
        return queries
    # makes entity type patterns less reliable        
    def adversial_noise(self, index, queries, entities, label, labels):
        should_add_noise = False
        inEntity = False
        secondCount = 0
        if index-3 < 0:
            return label
        for curr_index in range(index-3, index+10):
            if curr_index < index and entities[curr_index] != "O":
                return label
            if curr_index == index:
                inEntity = True
                continue
            if curr_index > index:
                if curr_index >= len(queries):
                    return label
                elif entities[curr_index] != "O":
                    if inEntity:
                        continue
                    else:
                        return label
                else:
                    inEntity = False
                    secondCount = secondCount + 1
                    if secondCount == 3:
                        should_add_noise = True
                        break
        labels.remove(label)
        curr = random.random()
        if should_add_noise and curr < self.threshold:
            print("Adding noise!")
            return random.choice(list(labels))
        return label
