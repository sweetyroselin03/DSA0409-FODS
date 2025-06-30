import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter

# Simple list of common English stop words
stop_words = set([
    'the', 'and', 'is', 'in', 'to', 'it', 'of', 'for', 'on', 'this', 'that', 'a',
    'an', 'with', 'as', 'was', 'but', 'are', 'at', 'by', 'be', 'or', 'from', 'has',
    'have', 'had', 'you', 'we', 'they', 'not', 'so', 'if', 'about', 'can', 'just',
    'do', 'out', 'my', 'your', 'our', 'more', 'too'
])

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words

def main():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("❌ Error: 'data.csv' not found.")
        return

    if 'feedback' not in df.columns:
        print("❌ Error: Column 'feedback' not found in the CSV.")
        return

    all_words = []
    for feedback in df['feedback'].dropna():
        all_words.extend(preprocess_text(str(feedback)))

    word_freq = Counter(all_words)

    try:
        N = int(input("Enter the number of top frequent words to display: "))
    except ValueError:
        print("❌ Error: Please enter a valid integer.")
        return

    top_words = word_freq.most_common(N)

    print(f"\nTop {N} Most Frequent Words:\n")
    for word, count in top_words:
        print(f"{word}: {count}")

    words, freqs = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs, color="skyblue")
    plt.title(f"Top {N} Most Frequent Words")
    plt.xlabel("Words")
    plt.ylabel("Frequencies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
