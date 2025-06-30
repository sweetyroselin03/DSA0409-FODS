import string
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Read the text file
with open('sample_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Step 2: Convert text to lowercase
text = text.lower()

# Step 3: Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Step 4: Tokenize the text (split into words)
words = text.split()

# Step 5: Count word frequencies
word_freq = Counter(words)

# Step 6: Display the top 10 most frequent words
print("📊 Top 10 Most Frequent Words:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")

# Step 7: Visualize the top 10 words using a bar chart
top_words = word_freq.most_common(10)
words, counts = zip(*top_words)

plt.figure(figsize=(8, 5))
plt.bar(words, counts, color='skyblue')
plt.title("Top 10 Word Frequencies")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
