from src.LDA.ModelBuilders.offline_model_builder import OfflineTraining
import pickle
from sklearn.metrics import classification_report
from src.CoNLLExtractor.preprocess import default_reading
ot = OfflineTraining("conll_data", noise=True)
model = ot.train()
ot.save("src/LDA/models/main-adv.dat")
data = default_reading("test")
query = data["queries"]
entities = data["named_entities"]
    
def load_model(model_filename):
    model_file = open(model_filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model

new_model = load_model("src/LDA/models/main-adv.dat")
#print(model.online_prediction(["former", "Leeds", "United", "defender"]))
predictions = []
for index, word in enumerate(query):
    max_length = 3
    length = 1
    sentence = [query[index][0]]
    # Adding to back and front so that the current word is centered
    # Made sure it was the same sentence
    while length < max_length/2 and (index-length) >= 0 and query[index][1] == query[index-length][1]:
        sentence = [query[index-length][0]] + sentence
        length = length + 1
    while length < max_length and (index+length) < len(query) and query[index][1] == query[index+length][1]:
        sentence.append(query[index+length][0])
        length = length + 1
    pred = new_model.online_prediction(sentence, model)
    print(pred)
    if len(pred) == 0:
        predictions.append("O")
    else:
        if word[0] not in pred[0]["triple"][0]:
            predictions.append("O")
        else:
            predictions.append(pred[0]["triple"][2])
#for index, prediction in enumerate(predictions):
#    if prediction.startswith("I-"):
#        if index-1 >= 0:
#            predictions[index-1] = prediction[2:]
#    if prediction.startswith("I-") or prediction.startswith("B-"):
#        predictions[index] = prediction[2:]
#for index, prediction in enumerate(entities):
#    if prediction.startswith("I-") or prediction.startswith("B-"):
#        entities[index] = prediction[2:]
print(entities)
print(predictions)
print(classification_report(entities, predictions))

