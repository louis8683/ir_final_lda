# Read vocabulary, document list, inverted-file
def read_vocab(filename, encoding='utf-8'):
    import codecs
    ''' Remember to open file f with the right encoding'''
    f = open(filename, "r", encoding=encoding)
    vocab = []
    for line in f:
        word = line.split('\n')[0]
        vocab.append(word)
    f.close()
    return vocab


def read_docs(filename):
    f = open(filename, "r")
    docs = []
    for line in f:
        doc = line.split('\n')[0]
        docs.append(doc)
    f.close()
    return docs


def read_inverted_file(filename):
    f = open(filename, "r")
    inverted_file_terms = []
    inverted_file_postings = []
    for line in f:
        line = line.split('\n')[0]
        vocab_id_1, vocab_id_2, n = line.split(' ')
        inverted_file_terms.append((int(vocab_id_1), int(vocab_id_2)))
        postings = []
        for _ in range(int(n)):
            line = f.readline()
            line = line.split('\n')[0]
            doc_id, cnt, tf_idf = line.split(' ')
            postings.append((int(doc_id), float(tf_idf)))
        inverted_file_postings.append(postings)
    f.close()
    return inverted_file_terms, inverted_file_postings
