# Read the HTML file
html_file_path = "words.html"
with open(html_file_path, "r") as file:
    html_content = file.read()

# Read the contents of words.json
json_file_path = "words.json"
with open(json_file_path, "r") as json_file:
    words_json = json_file.read()

json_file_path = "reduced_embeddings.json"
with open(json_file_path, "r") as json_file:
    embeddings_json = json_file.read()

# Replace "var words;" with "var words = " + contents of words.json
replacement_string = 'var words = ' + words_json + ';'
html_content = html_content.replace('var words;', replacement_string)

replacement_string = 'var embeddings = ' + embeddings_json + ';'
html_content = html_content.replace('var embeddings;', replacement_string)

# Write the modified HTML content to output.html
output_file_path = "output.html"
with open(output_file_path, "w") as file:
    file.write(html_content)

print("Replacement complete. Modified HTML saved to output.html.")
