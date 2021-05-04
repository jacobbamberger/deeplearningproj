import torch

def big_transform_to_one_hot(train): #this transformthe labels (0 to 9) into one hot vectors (with a one at index label)
    one_hot = torch.zeros((len(train),2, 10))
    for i in range(len(train)):
        one_hot[i, 0, train[i, 0]] = 1
        one_hot[i, 1, train[i, 1]] = 1
    return one_hot