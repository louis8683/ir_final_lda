import preprocessing as pp
import scipy.sparse as sparse
from sklearn.decomposition import LatentDirichletAllocation as LDA
from matplotlib import pyplot as plt
import pickle
import csv

'''
Parameters
'''

topics = 100
iterations = [256]
# iterations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

'''
Constants
'''

vocab_path = "./output_model/doc_200_term_20_model/vocal_doc_200"
docs_path = "./output_model/doc_200_term_20_model/file_list_doc_200"
invfile_path = "./output_model/doc_200_term_20_model/inverted_file_doc_200"
record_csv_filename = "record_final.csv"
lda_model_path = "./lda_final.pickle"
topic_words_filename_no_extension = "topic_words_"

def read_models():
    print("Reading models...")
    # Read files
    vocab = pp.read_vocab(vocab_path)
    docs = pp.read_docs(docs_path)
    # invf_terms: dim [][2]
    invf_terms, invf_postings = pp.read_inverted_file(invfile_path)
    return vocab, docs, invf_terms, invf_postings

def createMatrix(docs, invf_terms, invf_postings):
    # Create training matrix M[doc][term_ind]
    print("Creating training matrix...")
    M = []
    for _ in range(len(docs)):
        M.append([0] * len(invf_terms))
    # Populate with tf-idf in the document
    for i, postings in enumerate(invf_postings):
        for posting in postings:
            if posting[0] % 2 == 0: # Training document
                doc_ind = posting[0] // 2
                M[doc_ind][i] = posting[1]
        print(f"\r{i} of {len(invf_postings)} terms processed", end = '')
    print(f"\r{i} of {len(invf_postings)} terms processed")
    # Use sparse matrix. (?) Which kind of sparse to use?
    M = sparse.csr_matrix(M)
    return M

def trainLDA(X, max_iter=32, method='online', n_jobs=-1):
    print("Training with LDA...")
    lda = LDA(n_components=topics, max_iter=max_iter, n_jobs=n_jobs, verbose=1, learning_method=method)
    lda.fit(X)
    return lda

def plotPerplexity(record):
    plt.plot(record)
    plt.xlabel("iterations (2^n)")
    plt.ylabel("perplexity")
    plt.show()

def multipleTrainLDA(X, iterations, record, method='online', n_jobs=-1, dump=False):
    for iteration in iterations:
        print(f"Iterations: {iteration}...")
        lda = trainLDA(M, iteration, method)
        perplexity = lda.perplexity(M)
        print(f"Perplexity (train): {perplexity}")
        record.append(perplexity)
        if dump:
            pickle.dump(lda, open(f"lda_iter{iteration}_{method}.pickle", "wb"))
    return lda

def saveRecord(iterations, record):
    with open(record_csv_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iterations", "perplexity"])
        for i in range(len(record)):
            writer.writerow([iterations[i], record[i]])

def save_LDA_model(lda):
    pickle.dump(lda, open(lda_model_path, "wb"))

'''
Read Models
'''

vocab, docs, invf_terms, invf_postings = read_models()

'''
Training
'''

M = createMatrix(docs, invf_terms, invf_postings)

record = []

lda = multipleTrainLDA(M, iterations, record)
if len(iterations > 1):
    plotPerplexity(record)

'''
Save Progress
'''

saveRecord(iterations, record)
save_LDA_model(lda)

'''
Split terms
'''

# Find words in each topic
topic_words = []
for _ in range(topics):
    topic_words.append([])
for term_ind in range(len(invf_terms)):
    print(f"\rProcessing {term_ind} of {len(invf_terms)} Terms...", end='')
    # Init term vector
    vec = [0] * len(invf_terms)
    vec[term_ind] = 1
    scores = lda.transform([vec])
    max_ind = 0
    max_score = 0
    for score_ind, score in enumerate(scores[0]):
        if score > max_score:
            max_ind = score_ind
            max_score = score
    topic_words[max_ind].append((max_score, term_ind))

# Sort each topic by score
for words in topic_words:
    words.sort()

# Save topic words verbally into separate files
for topic, words in enumerate(topic_words):
    with open(f"{topic_words_filename_no_extension}{topic}.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for term_ind in words:
            term = invf_terms[term_ind[1]]
            word1, word2 = vocab[term[0]], vocab[term[1]]
            writer.writerow([word1 + word2])
