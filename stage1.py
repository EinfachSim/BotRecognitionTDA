import numpy as np
import pandas as pd
import networkx as nx
import gtda as tda
import math

USERS_PATH = "../dataset/preprocessed/filtered_users.csv"
EDGES_PATH = "../dataset/preprocessed/filtered_edges.csv"
#returns the distance matrix for the user's ego network specified in user_id.
#needs some adjustment to return not only the adjacency matrix but
#the weighted adj_matrix
#returned np.array should be sparse
def get_distance_matrix(user_id: str, G: nx.DiGraph) -> np.ndarray:
    print(f"Getting Ego Network for user {user_id}...")
    ego = nx.ego_graph(G, user_id, radius=1, undirected=True)
    """
    TODO:
        [x] weight graph according to mca paper
        [] get some statistics, need to read Felix' and Christian's paper to decide which one.
    """
    #update the weights according to mca paper
    ego = get_ego_undirected_weighted(G, ego)
    print(nx.adjacency_matrix(ego).toarray())
    # toarray() because nx returns a scipy sparse matrix, but giotto (TDA)
    # assumes a np.ndarray
    print("Done!")
    return nx.adjacency_matrix(ego).toarray()
"""
This method computed the persistence diagrams obtained by WeightedVietorisRips
filtration (and maybe others).
takes in one ego network as np.ndarray (adjacency/distance matrix)
returns a list of np.ndarrays of shape (1,n_features, 3)
each entry in that list corresponds to a persistence diagram in the given homology
dimension
"""
def get_persistence_diagram(ego: np.ndarray) -> np.ndarray:
    WVR = tda.WeightedRipsPersistence(metric="precomputed")

"""
This method is used to get the degrees of strength per edge as defined in mca.
Takes in a graph G and an ego network from a node in the graph G
Returns a dictionary with keys being the edges and values being the degree of strength,
the mean of the degrees of strength and the standard deviation of the degrees of strength
"""
def get_degrees_of_strength(G: nx.DiGraph, ego: nx.DiGraph) -> tuple:
    s = {}
    for e in ego.edges:
        A = e[0]
        B = e[1]
        B_pre = G.in_degree(B)
        A_suc = G.out_degree(A)
        degree_of_strength = math.log(B_pre/A_suc)
        s[e] = degree_of_strength
    return s, np.mean(list(s.values())), np.std(list(s.values()))

"""
This method computes the weights for the edges in the ego network and normalizes them
by their mean and std (specifically by mean and std of the degrees of strength, see
get_degrees_of_strength)
Takes in a graph G and an ego network from a node in G and optionally a scaling parameter L
default L = 100
Returns an nx.Graph with the edge weights being as seen in the mca paper
"""
def get_ego_undirected_weighted(G, ego_nw, L=1):
    ego = ego_nw.copy()
    #Get degrees of strength
    s, mean, std = get_degrees_of_strength(G, ego_nw)
    #Compute weights for every edge
    weights = {}
    for e in ego.edges:
        w = L * (1/(1+np.exp((-s[e] + mean)/std)))
        ego[e[0]][e[1]]["weight"] = w
    #Setting up for to_undirected
    for e in ego.edges:
        w1 = ego[e[0]][e[1]]["weight"]
        if(not ego.has_edge(e[1], e[0])):
            ego.add_edge(e[1], e[0], weight=w1)
        else:
            w2 = ego[e[1]][e[0]]["weight"]
            actual_weight = max(w1, w2)
            ego[e[0]][e[1]]["weight"] = actual_weight
            ego[e[1]][e[0]]["weight"] = actual_weight
    return ego.to_undirected()

if __name__ == "__main__":
    users_df = pd.read_csv(USERS_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    print("Building Graph...")
    G = nx.from_pandas_edgelist(edges_df, "source_id", "target_id", create_using=nx.DiGraph())
    print("Done!")
    #Just for testing purposes:
    ego_nw = get_distance_matrix("u1484424461915127810", G)
    #print(get_persistence_diagram(ego_nw))