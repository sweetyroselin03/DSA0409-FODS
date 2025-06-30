import pandas as pd
from collections import Counter
import string

# New dataset
data = {
    'ReviewID': [101, 102, 103, 104],
    'ReviewText': [
        "Excellent service and fast delivery!",
        "Poor packaging, but the product was good.",
        "Value for money. Will buy again.",
        "Product arrived late and damaged."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Process text: lowercase, remove punctuation, split into words
all_reviews = ' '.join(df['ReviewText'].str.lower())
all_reviews = all_reviews.translate(str.maketrans('', '', string.punctuation))
words = all_reviews.split()

# Count word frequencies
word_freq = Counter(words)

# Display frequency distribution
print("Frequency distribution of words in customer reviews:")
for word, freq in word_freq.items():
    print(f"{word}: {freq}")
