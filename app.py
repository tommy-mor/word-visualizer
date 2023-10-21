oxford_3000 = []
with open("words.txt", 'r') as f:
    oxford_3000 = [w.strip() for w in f.readlines()]

oxford_3000[4]

#import openai


# read secret from secrets.json
#import json
#with open("secrets.json") as f:
    #secrets = json.load(f)
    
#openai.api_key = secrets["api_key"]

#word_embeddings_map = {}
#
#model = "text-embedding-ada-002"
#n_items = len(oxford_3000)
#batch_size = 1000
#n_batches = (n_items + batch_size - 1) // batch_size
#for i in range(n_batches):
    #start,end = i * batch_size, (i + 1) * batch_size
    #inp = oxford_3000[start:end]
    #print(start,end)
    #response = openai.Embedding.create(input=inp, model=model)
    #embeddings = [i["embedding"] for i in response["data"]]
    #for word, embedding in zip(inp, embeddings):
        #word_embeddings_map[word] = embedding

import pickle
#with open('embeddings.pickle', 'ab') as f:
    #pickle.dump(word_embeddings_map, f)
with open('embeddings.pickle', 'rb') as f:
    word_embeddings_map = pickle.load(f)

import numpy as np
#from sklearn.decomposition import PCA
#import plotly.express as px

words = np.array(list(word_embeddings_map.keys()))

#embeddings = []
#for w in words:
    #embeddings.append(word_embeddings_map[w])
#embeddings = np.array(embeddings)
embeddings = np.array(list(word_embeddings_map.values()))

def find_embedding(word):
    print(word)
    index = np.where(words == word)[0][0]
    return embeddings[index]

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/embeddings')
def embeddings_json():
    x1 = request.args.get('x1')
    x2 = request.args.get('x2')
    y1 = request.args.get('y1')
    y2 = request.args.get('y2')
    z1 = request.args.get('z1')
    z2 = request.args.get('z2')


    x = find_embedding(x1) - find_embedding(x2)
    y = find_embedding(y1) - find_embedding(y2)
    z = find_embedding(z1) - find_embedding(z2)
    
    xs = np.dot(embeddings, x).tolist()
    ys = np.dot(embeddings, y).tolist()
    zs = np.dot(embeddings, z).tolist()

    print(xs[0])

    return {"xs": xs, "ys": ys, "zs": zs, "words": words.tolist()}

@app.route("/")
def index():
    with open("change.html") as f:
        return f.read()
    
@app.route('/has_word')
def has_word():
    word = request.args.get('word')

    found = False

    try:
        find_embedding(word).tolist()
        found = True
    except:
        found = False
    
    return {"found":  found}

if __name__ == '__main__':
    app.run(debug=True)
