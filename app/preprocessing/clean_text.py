import re
import unicodedata
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag





import torch
# Load English stop words
stop_words = set(stopwords.words('english'))

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to wordnet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def replace_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def is_valid_utf8(text):
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    else:
        return True

def clean_text(text):
    # Decode HTML entities
    text = html.unescape(text)
    
    # Normalize unicode characters to ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Convert text to lowercase to standardize it
    text = text.lower()
    
    # Remove HTML tags using a regular expression
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs using a regular expression
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = replace_multiple_spaces(text)
    
    # Check for valid UTF-8
    if not is_valid_utf8(text):
        return ""
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    filtered_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tag(tokens)
        if word not in stop_words and len(word) > 1
    ]
    
    # Reconstruct the text without stop words
    return ' '.join(filtered_tokens)


# def get_tweet_embeddings(tweet,
#                          tokenizer, 
#                          device, 
#                          bert_model,
#                          max_length=34):

#     # Tokenize the batch of tweets
#     encoded_inputs = tokenizer(tweet, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
#     # encoded_input = {k: v.to(device) for k, v in encoded_inputs.items()}
#     return encoded_inputs
    # Get the hidden states from the BERT model
    # with torch.no_grad():
    #     outputs = bert_model(**encoded_inputs)
    #     hidden_states = outputs.last_hidden_state

    
    # for j in range(hidden_states.size(0)):
    #     # Iterate over each tweet in the batch
    #     subword_embeddings = []
    #     tweet_embedding = []
    #     for k in range(hidden_states.size(1)):
    #         # Iterate over each token in the tweet
    #         if encoded_inputs['attention_mask'][j][k] == 1:
    #             # Check if the token is not a padding token
    #             word_embedding = hidden_states[j][k]

    #             if tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][j][k].item()).startswith('##'):
    #                 # If the token is a subword, append its embedding to the subword_embeddings list
    #                 subword_embeddings.append(word_embedding)
    #             else:
    #                 if subword_embeddings:
    #                     # If there are previously collected subword embeddings, average them
    #                     averaged_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
    #                     tweet_embedding.append(averaged_embedding)
    #                     subword_embeddings = []

    #             # Append the word embedding to the tweet_embedding list
    #             tweet_embedding.append(word_embedding)

    #     if subword_embeddings:
    #         # If there are any remaining subword embeddings, average them and append to the tweet_embedding list
    #         averaged_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
    #         tweet_embedding.append(averaged_embedding)

    #     # Pad the word embeddings to the maximum length
    #     padding = [torch.zeros(hidden_states.size(2)).to(device) for _ in range(max_length - len(tweet_embedding))]
    #     tweet_embedding.extend(padding)
    #     tweet_embedding = torch.stack(tweet_embedding)
