#!/usr/bin/env python3
import numpy as np

def entropy(prob): 
    prob = prob[prob != 0]
    return np.sum(prob * -np.log(prob))
    
def cross_entropy(data, model): 
    assert len(data) == len(model)
    not_zero = data != 0
    data = data[not_zero]
    model = model[not_zero]
    with np.errstate(divide='ignore'):
        return np.sum(data * -np.log(model))

def kl_divergence(data, model): 
    assert len(data) == len(model)
    not_zero = data != 0 
    data = data[not_zero]
    model = model[not_zero]
    with np.errstate(divide='ignore'):
        return np.sum(data * (np.log(data) - np.log(model)))


    
if __name__ == "__main__":
    # Load data distribution, each data point on a line
    d = []
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            d.append(line)
            
    # Load model distribution, each line `word \t probability`.
    m = {}
    m_ids = []

    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            line = line.split("\t")
            assert len(line) == 2
            assert not(line[0] in m)
            m_ids.append(line[0])
            m[line[0]] = float(line[1])
    
    ids = sorted((set(m_ids) | set(d))) 
    k, c = np.unique(d, return_counts=True)
    d = dict(zip(k, c/sum(c)))
    d = np.array([d.get(x,0) for x in ids]) 
    m = np.array([m.get(x,0) for x in ids])
    
    # TODO: Create a NumPy array containing the model distribution.
    
    # TODO: Assert the indexes
    
    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    print("{:.2f}".format(entropy(d)))
    print("{:.2f}".format(cross_entropy(d,m)))
    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    print("{:.2f}".format(kl_divergence(d,m)))