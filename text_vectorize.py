import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

""" the concept is to create a bag of wards that appears in the mails 
    i removed the words that appears few time and removed the words that have no meaning"""
def vectorize(df):
    remove_from_data = ['{', '}', '(', ')', '/', ',', '.', ';', ':', '!', '?', '-', '_', '=', '+', '[', ']',
        '"', "'", 'subject:', 'the', 'a', 'this', 'of', 'and', 'to', 'in', 'is', 'that', 'for',
        'as', 'by', 'be', 'it', 'with', 'on', 'was', 'or', 'not', 'i', 'are', 'from', 'at',
        'which', 'but', 'if', 'you', 'can', 'have', 'has', 'had', 'will', 'would', 'should',
        'could', 'an', 'we', 'he', 'she', 'they', 'them', 'their', 'there', 'then', 'than',
        'been', 'being', 'so', 'some', 'such', 'no', 'nor', 'my', 'your', 'its', 'about',
        'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'did', 'do', 'does', 'done',
        'these', 'those', 'am', 'me', 'him', 'her', 'ours', 'ourselves', 'yours', 'yourselves',
        'themselves', 'this', 'that', 'these', 'those', 'each', 'every', 'other', 'another',
        'own', 'same', 'such', 'than', 'too', 'very','@']
    text = df["text"].str.lower().str.split()
    text = text.apply(lambda words: [word for word in words if word not in remove_from_data])
    set_of_words = create_list_of_words(text)
    vectors = create_data(text,set_of_words)
    vectors = apply_pca(vectors)
    length = lenoftext(text)
    x=pd.DataFrame(vectors)
    x[100] = pd.Series(length)
    return x
"create the bag of wards"
def create_list_of_words (texts,minimum_apr = 5):
    dict_of_words = {}
    for list in texts:
        for word in list:
            dict_of_words[word] = dict_of_words.get(word,0)+1
    res= [word for word in dict_of_words.keys() if dict_of_words[word]>minimum_apr]
    return res
""" create the vectors for each mail """
def create_data(texts,set):
    vectors = []
    set_of_words_list = list(set)
    word_to_index = {word: idx for idx, word in enumerate(set_of_words_list)}
    num_features = len(set_of_words_list)

    for words in texts:
        word_counter = np.zeros(num_features, dtype=int)
        for word in words:
            if word in word_to_index:
                word_counter[word_to_index[word]] += 1
        vectors.append(word_counter)

    return np.array(vectors)
""" to reducte the number of features 
    I tried without the pca and recived overfittingv traun set score was 1.0"""
def apply_pca(vectors, n_components=430):
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    return reduced_vectors

def lenoftext(texts):
    res = [len(text) for text in texts]
    return res









