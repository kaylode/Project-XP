import string
import nltk
import json



def process_data(*paths):
    datadict = {}
    for path in list(paths):
        print(path)
        assert type(path)==type("str"), "path not right"
        with open(path, "r") as f:
            json_data = json.load(f)
            images_list = json_data["images"]
            annotations_list = json_data["annotations"]   

        for key in images_list:
            datadict[key["id"]] = {"file_name": key["file_name"], "captions": []}
        for anno in annotations_list:
            datadict[anno["image_id"]]["captions"].append(anno["caption"])
    return datadict

def preprocessing(sentence):
   
    def word_lowercase(sentence):
        return sentence.lower()
    
    def clean(sentence):
        sentence = word_lowercase(sentence)
        return sentence
    
    sentence = clean(sentence)
    return sentence

def mytokenizer(sentence):
    sentence = preprocessing(sentence)
    tokens = nltk.tokenize.word_tokenize(sentence)
    return tokens