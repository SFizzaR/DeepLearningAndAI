import string
from nltk.corpus import stopwords
import nltk 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import numpy

#Task 1: Preprocessing the Text
def createTokens(corpus):
    tokens= corpus.split()
    stop_words = set(stopwords.words('english'))
    clean_tokens = []
    for token in tokens: 
        cleaned = token.strip(string.punctuation).lower()
        if cleaned and cleaned not in stop_words: 
            clean_tokens.append(cleaned)
    return clean_tokens

def create_mappings(tokens):
    unique_words = sorted(set(tokens))

    word2idx = {word: idx for idx, word in enumerate(unique_words)}
    idxtowords = {idx: word for word, idx in word2idx.items()}

    return word2idx, idxtowords

def generate_skipgram_pairs(tokens, window_size=2):
    skipgram_pairs = []

    for idx, target_word in enumerate(tokens):
        start_idx = max(0, idx-window_size)
        end_idx = min(len(tokens), idx+window_size+1)

        for context_idx in range(start_idx, end_idx):
            if context_idx != idx: 
                context_word = tokens[context_idx]
                skipgram_pairs.append((target_word, context_word))

    return skipgram_pairs

#Task 2: Implementing the Skip-gram Model
class SkipgramModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipgramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, target_word):
        target_embedding = self.embedding(target_word)
        output = self.fc(target_embedding)

        return output

def prepare_skipgram_training_data(skipgram_pairs, word2idx):
    training_data = []
    for target_word, context_word in skipgram_pairs:
        target_idx = word2idx[target_word]
        context_idx = word2idx[context_word]
        training_data.append((target_idx, context_idx))
    return training_data

#Task 3: Training the Model
def train_skipgram_model(training_data, vocab_size, embedding_size):
    model = SkipgramModel(vocab_size, embedding_size)
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)

        for target_idx, context_idx in training_data:
            target_tensor = torch.tensor([target_idx])
            context_tensor = torch.tensor([context_idx])

            output= model(target_tensor)

            loss = criterion(output, context_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            if (epoch + 1)%10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model
    

def save_embeddings(model, idx2word, filename="word_embeddings.txt"):
    embeddings = model.embedding.weight.data
    with open(filename, "w") as f:
        for idx, vector in enumerate(embeddings):
            word = idx2word[idx] 
            vector_str = ' '.join([f"{val:.4f}" for val in vector.tolist()])
            f.write(f"{word} {vector_str}\n")
    print(f"Embeddings saved to file {filename}")

#Task 4: Visualizing Word Embeddings
def visualize_embeddings(model, idx2word):

    embeddings= model.embedding.weight.data.cpu().numpy()
    tsne = TSNE(n_components=2, random_state= 0, perplexity=5)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for idx, coord in enumerate(reduced_embeddings):
        word = idx2word[idx]
        x, y = coord
        plt.scatter(x, y)
        plt.annotate(word, (x+0.01, y+0.01)) #slight offset to prevent overlap 
    
    plt.title("Word Embeddings Visualized with t_SNE")
    plt.grid(True)
    plt.show()

#Task 5
def generate_cbow_pairs(tokens, window_size=2):
    cbow_pairs = []
    for idx in range(window_size, len(tokens)-window_size):
        context =[]
        for i in range (-window_size, window_size+1):
            if i != 0:
                context.append(tokens[idx+i])
        cbow_pairs.append((context, tokens[idx]))
    return cbow_pairs

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, context_word_idx):
        embeddings = self.embedding(context_word_idx)
        mean_embedding = embeddings.mean(dim=0)
        output = self.fc(mean_embedding)
        return output 
    
def prepare_cbow_training_data(cbow_pairs, word2idx):
    training_data = []
    for context_words, target_word in cbow_pairs:
        context_idxs = [word2idx[word] for word in context_words]
        target_idx = word2idx[target_word]
        training_data.append((context_idxs, target_idx))
    return training_data

def train_cbow_model(training_data, vocab_size, embedding_size):
    model = CBOWModel(vocab_size, embedding_size)
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)

        for context_idxs, target_idx in training_data:
            target_tensor = torch.tensor([target_idx])
            context_tensor = torch.tensor(context_idxs)

            output= model(context_tensor)

            loss = criterion(output.unsqueeze(0), target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            if (epoch + 1)%10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model

def findanalogy(word_a, word_b, word_c, model, idx2word, word2idx):
    emb = model.embedding.weight.data  
    
    try:
        vec = emb[word2idx[word_a]] - emb[word2idx[word_b]] + emb[word2idx[word_c]]
    except KeyError as e:
        return f"Word not in vocabulary: {e}"
    
    similarities = torch.matmul(emb, vec)
    most_similar_idx = torch.argmax(similarities).item()

    return idx2word[most_similar_idx]

vocab_size = 11 
embedding_size = 10

if __name__ == "__main__":
    with open('text_corpus.txt', 'r') as file:
        content = file.read()

    tokens = createTokens(content)
    w2i, i2w = create_mappings(tokens)
    skipgram_pairs = generate_skipgram_pairs(tokens)
    skipgram_training_data = prepare_skipgram_training_data(skipgram_pairs, w2i)

    vocab_size = len(w2i)
    embedding_size = 10
    skipgram_model = train_skipgram_model(skipgram_training_data, vocab_size, embedding_size)
    save_embeddings(skipgram_model, i2w)

    visualize_embeddings(skipgram_model, i2w)

    #CBOW
    cbow_pairs = generate_cbow_pairs(tokens)
    cbow_training_data = prepare_cbow_training_data(cbow_pairs, word2idx=w2i)
    cbow_model = train_cbow_model(cbow_training_data, vocab_size, embedding_size)
    result = findanalogy("ai", "computers", "learning", cbow_model, i2w, w2i)
    print(f"ai - computers + learning ≈ {result}")

    

