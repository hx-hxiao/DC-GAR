import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import networkx as nx
from node2vec import Node2Vec
import torch
from torch_geometric.data import Data
import time
import torch.nn as nn

start_time = time.time()


def create_graph_classification_dataset(doc_dir, label_file):
    dataset = []

    with open(label_file, "r") as f:
        labels = [line.strip() for line in f]

    for i, filename in enumerate(sorted(os.listdir(doc_dir))):
        graph_path = os.path.join(doc_dir, filename)
        g = nx.drawing.nx_pydot.read_dot(graph_path)
        G = nx.DiGraph()
        G.add_nodes_from(g.nodes())
        G.add_edges_from(g.edges())

        def create_input_features(G):
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            avg_degree = 2 * num_edges / num_nodes
            clustering_coef = nx.average_clustering(G)
            # list
            cc = list(nx.clustering(G).values())
            pagerank = list(nx.pagerank(G).values())
            dc = list(nx.degree_centrality(G).values())
            closen = list(nx.closeness_centrality(G).values())
            bc = list(nx.betweenness_centrality(G).values())
            node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=30, workers=4, p=2, q=0.5)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            embeddings = model.wv

            node_features = []
            for node in G.nodes:
                if str(node) in embeddings:
                    node_features.append(embeddings[str(node)])
                else:
                    node_features.append(np.zeros(11))
            node_features = torch.tensor(node_features, dtype=torch.float)
            print(node_features)

            features = np.column_stack((
                pagerank,
                cc,
                dc,
                closen,
                bc,
                node_features
            ))

            scaler = StandardScaler()
            X = scaler.fit_transform(features)
            return X

        X = create_input_features(G)
        X = torch.Tensor(X)

        node_ids = {node: i for i, node in enumerate(G.nodes)}
        edge_index = (torch.tensor([[node_ids[u], node_ids[v]] for u, v in G.edges()], dtype=torch.long)
                      .t().contiguous())

        data = Data(x=X, edge_index=edge_index, y=torch.tensor([int(labels[i])], dtype=torch.long))
        dataset.append(data)

    return dataset


if __name__ == '__main__':
    main()


end_time = time.time()
run_time = end_time - start_time  
print(f"Time: {run_time:.2f} s")
