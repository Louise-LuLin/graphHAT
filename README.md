# Requirement
* `Python 3.6`
* `Tensoflow 1.14`

# Dataset Preparation
First, process dataset to generate the graph and node features. The format is as follows:
* `feature.bin`, the feature vectors of the node instances. To save the space, we store them as sparse matrix via scipy.sparse.lil.lil_matrix. Each row represents all the feature of a node.
* `label.bin`, the one-hot labels of the node instances. Each row represents a one-hot vector of class label.
* `graph.bin`, a dict in the format {index: [index_of_neighbor_nodes]}, where the neighbor nodes are organized as a list.

Then, we will randomly sample nodes to split the nodes into train/eval/test sets. For citation datasets `cora, citeseer, pubmed`, we follow the processing of [GAT](https://github.com/PetarV-/GAT/blob/master/utils/process.py) to split train/val/test sets. For amazon product dataset, we will use the default data separation given by [OGB](https://ogb.stanford.edu/docs/nodeprop/#loader). For other datasets, we will take the ratio=60%, 20%, 20%. The node indexs are stored as:
* `idx_train.bin`, the indexs of training instances.
* `idx_eval.bin`, the indexs of evaluation instances.
* `idx_test.bin`, the indexs of test instances.

Please refer to `test_data.ipynb` for the detailed information of the datasets under `./data` folder.

