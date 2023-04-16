# Homework #1-1
# Problem 1
# Calculate cosine similarity given 2 sentence strings

# import numpy for vector operations
import numpy as np

# define the sentences
s1 = "try not to become a man of success but rather try to become a man of value"
s2 = "theories should be as simple as possible, but not simpler"

# split the sentences into words
words1 = s1.split()
words2 = s2.split()

# create a set of unique words from both sentences
vocab = set(words1 + words2)

# create a dictionary to map each word to an index
word_to_index = {w: i for i, w in enumerate(vocab)}
# create two zero vectors of the same length as the vocabulary
v1 = np.zeros(len(vocab))
v2 = np.zeros(len(vocab))

# iterate through the words of the first sentence and increment the corresponding index in the vector
for w in words1:
    v1[word_to_index[w]] += 1

# iterate through the words of the second sentence and increment the corresponding index in the vector
for w in words2:
    v2[word_to_index[w]] += 1

# calculate the dot product of the two vectors
dot_product = np.dot(v1, v2)

# calculate the magnitude of each vector
v1_magnitude = np.sqrt(np.sum(v1**2))
v2_magnitude = np.sqrt(np.sum(v2**2))

# calculate the cosine similarity by dividing the dot product by the product of the magnitudes
cosine_sim = dot_product / (v1_magnitude * v2_magnitude)

# print the result
print(cosine_sim)
