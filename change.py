oxford_3000 = []
with open("words.txt", 'r') as f:
    oxford_3000 = [w.strip() for w in f.readlines()]

oxford_3000[4]

import openai
openai.api_key = "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"

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
from sklearn.decomposition import PCA
import plotly.express as px

words = np.array(list(word_embeddings_map.keys()))

#embeddings = []
#for w in words:
    #embeddings.append(word_embeddings_map[w])
#embeddings = np.array(embeddings)
embeddings = np.array(list(word_embeddings_map.values()))


axis = [["small", "large"],
        ["mean", "happy"],
        ["sweet", "bitter"]]

def find_embedding(word):
    print(word)
    index = np.where(words == word)[0][0]
    return embeddings[index]

#x = find_embedding(axis[0][0]) - find_embedding(axis[0][1])
#y = find_embedding(axis[1][0]) - find_embedding(axis[1][1])
#z = find_embedding(axis[2][0]) - find_embedding(axis[2][1])
#
#xs = np.dot(embeddings, x)
#ys = np.dot(embeddings, y)
#zs = np.dot(embeddings, z)


# find the vectors for each word
# one dim is small - large, then each word is that dim cross product



import plotly.graph_objects as go

def visualize_3d_highlight(words_to_highlight):

    # Default values for all words
    colors = ['blue' if word not in words_to_highlight else 'red' for word in words]
    opacities = [0.1 if word not in words_to_highlight else 1.0 for word in words]
    sizes = [5 if word not in words_to_highlight else 10 for word in words]
    texts = [word if word not in words_to_highlight else word for word in words]

    # Create the scatter plot
    scatter = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(color=colors, size=sizes),
        text=texts
    )
    layout = go.Layout(height=800)  # Set height as per your preference
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

# Call the function and pass the words you want to highlight
words_to_highlight = ["life", "death", "awake", "asleep", "day", "night"] # replace with the words you want
#visualize_3d_highlight(words_to_highlight)

#import json
#with open('reduced_embeddings.json', 'w') as f:
    #json.dump(reduced_embeddings.tolist(), f)
#
#with open('words.json', 'w') as f:
    #json.dump(words, f)

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
