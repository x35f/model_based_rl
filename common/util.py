from operator import itemgetter
import random
import numpy as np
import torch

def ensemble_sampler(data, ensemble_size, batch_size):
    keys = list(data.keys())
    data_items = itemgetter(*keys)(data)
    data_size = len(data_items[0])
    sample_indices = [list(range(data_size)) for i in range(ensemble_size)]
    for i in range(ensemble_size):
        random.shuffle(sample_indices[i])
    num_batches = int(np.ceil(data_size/batch_size))
    for batch_idx in range(num_batches):
        curr_data ={}
        batch_data_indices = [indices[batch_idx * batch_size: min((batch_idx + 1) * batch_size, data_size)] for indices in sample_indices]

        for key in keys:
            curr_data[key] = np.stack([data[key][batch_data_indices[i]]
             for i in range(ensemble_size)])
        yield curr_data


def minibatch_rollout(model, input_data, mini_batch_size):
    data_size = len(input_data)
    outputs = []
    num_rollouts = int(np.ceil(data_size / mini_batch_size))
    for i in range(num_rollouts):
        input_data_batch = input_data[i * mini_batch_size, min(data_size, (i + 1) * mini_batch_size)]
        outputs.append(model(input_data_batch)) 
    return torch.stack(outputs)




if __name__ == "__main__":
    data = {"obs": np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]]), "r": np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]])}
    for d in ensemble_sampler(data, 3,3):
        print(d)