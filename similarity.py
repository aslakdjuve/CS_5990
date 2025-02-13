# -------------------------------------------------------------------------
# AUTHOR: Aslak Djuve
# FILENAME: Question 8
# SPECIFICATION: 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 30 min
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)
         print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
docTermMatrix = []
uniqueWords = set()

# First pass - collect all unique words
for doc in documents:
    words = doc[1].split() # doc[1] contains the text
    uniqueWords.update(words)

uniqueWords = list(uniqueWords) # Convert to list for easier indexing

#Binary encoding
for doc in documents:
    vector = [0] * len(uniqueWords)
    words = doc[1].split()
    for word in words:
        if word in uniqueWords:
            vector[uniqueWords.index(word)] = 1
    docTermMatrix.append(vector)

# Compare pairwise cosine similarities
similarity_matrix = cosine_similarity(docTermMatrix)
max_sim = -1.0
best_i = -1
best_j = -1

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > max_sim:
            max_sim = similarity_matrix[i][j]
            best_i = i
            best_j = j


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are document {documents[best_i][0]} and document {documents[best_j][0]} with cosine similarity = {max_sim:.4f}.")

