# Requirement
* `Python 3.6`
* `Tensoflow 1.14`

# dataset preparation
First, process dataset into the format as follows:
* `allx`, the feature vectors of the training instances, 
* `ally`, the one-hot labels of the training instances,
* `graph`, a dict in the format {index: [index_of_neighbor_nodes]}, where the neighbor nodes are organized as a list.

Then 

For citation datasets `cora, citeseer, pubmed`, we follow the processing of [GAT](https://github.com/PetarV-/GAT/blob/master/utils/process.py) to split train/val/test sets.
