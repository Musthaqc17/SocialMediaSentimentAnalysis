import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLP resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize NLP tools
STOP_WORDS = set(stopwords.words("english"))

# Preserve negations because they affect sentiment
STOP_WORDS.discard("not")
STOP_WORDS.discard("no")
STOP_WORDS.discard("nor")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean social media text for sentiment analysis.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Preserve hashtag words while removing the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    Tokenize cleaned text and remove stopwords.
    """

    cleaned = clean_text(text)

    if not cleaned:
        return []

    tokens = word_tokenize(cleaned)

    return [
        word
        for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]


def stem_text(text):
    """
    Apply stemming to cleaned text.
    """

    tokens = tokenize_text(text)

    return " ".join(stemmer.stem(word) for word in tokens)


def lemmatize_text(text):
    """
    Apply lemmatization to cleaned text.
    """

    tokens = tokenize_text(text)

    return " ".join(
        lemmatizer.lemmatize(word)
        for word in tokens
    )


if __name__ == "__main__":

    sample = "I am NOT happy with this flight! The service was terrible. #BadExperience"

    print("Original text:")
    print(sample)

    print("\nCleaned text:")
    print(clean_text(sample))

    print("\nTokens:")
    print(tokenize_text(sample))

    print("\nStemmed text:")
    print(stem_text(sample))

    print("\nLemmatized text:")
    print(lemmatize_text(sample))
