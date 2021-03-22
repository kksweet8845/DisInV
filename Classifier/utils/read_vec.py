import pickle
import torch
import numpy as np
from os import path
import zipfile


def vec2dat(filename):

    word_dict = {}
    with open(filename, 'r') as file:
        first = list(map(int, file.readline().split()))
        vocab = first[0]
        dim = first[1]

        for i in range(vocab):
            line = file.readline().split()
            word = " ".join(line[0:len(line) - dim])

            vector = list(map(float, line[len(line)-dim:]))
            word_dict[word] = vector

    
    pickle.dump(word_dict, open('../model_src/news-vec-final.dat', 'wb'))

    test = word_dict['餐廳']

    word_dict = pickle.load(open('../model_src/news-vec-final.dat', 'rb'))

    for i in range(300):
        assert word_dict['餐廳'][i] == test[i], "wrong vector"



def vec2Tensor(filename, output_dir):

    word_dict = {}
    ls = []
    with open(filename, 'r') as file:
        first = list(map(int, file.readline().split()))
        vocab = first[0]
        dim = first[1]

        for i in range(vocab):
            line = file.readline().split()
            word = " ".join(line[0:len(line) - dim])

            vector = list(map(float, line[len(line)-dim:]))
            word_dict[word] = i
            ls.append(vector)

    vec_tensor = torch.tensor(ls, dtype=torch.float)

    with open(path.join(output_dir, "embedding.ts"), 'wb') as file:
        pickle.dump(vec_tensor, file)

    with open(path.join(output_dir, "lookup.tb"), 'wb') as file:
        pickle.dump(word_dict, file)


    # test
    with open(path.join(output_dir, "embedding.ts"), 'rb') as file:
        embedding = pickle.load(file)

    with open(path.join(output_dir, "lookup.tb"), 'rb') as file:
        lookup_tb = pickle.load(file)

    assert torch.equal(vec_tensor, embedding) == True, "Not equal"

def vec2Tensor_nlpl(filename, output_dir):

    word_dict = {}
    ls = []

    with zipfile.ZipFile(filename) as archive:
        stream = archive.open('model.txt')

        first = list( map(int, stream.readline().split()))
        vocab = first[0]
        dim = first[1]

        for i in range(vocab):
            line = stream.readline().decode('utf-8').split()
            word = " ".join(line[0:len(line) - dim])

            vector = list(map(float, line[len(line)-dim:]))
            word_dict[word] = i
            ls.append(vector)

    vec_tensor = torch.tensor(ls, dtype=torch.float)

    with open(path.join(output_dir, "embedding.ts"), 'wb') as file:
        pickle.dump(vec_tensor, file)

    with open(path.join(output_dir, "lookup.tb"), 'wb') as file:
        pickle.dump(word_dict, file)


    # test
    with open(path.join(output_dir, "embedding.ts"), 'rb') as file:
        embedding = pickle.load(file)

    with open(path.join(output_dir, "lookup.tb"), 'rb') as file:
        lookup_tb = pickle.load(file)

    assert torch.equal(vec_tensor, embedding) == True, "Not equal"

if __name__ == '__main__':
    # pass
    # vec2dat("../model_src/news-corpus-final.vec")
    # vec2Tensor("../model_src/news-corpus-final.vec", "../model_src")
    # vec2Tensor_nlpl("../vectors/9.zip", "../model_src-ENG")
    pass