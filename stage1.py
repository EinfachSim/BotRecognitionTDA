import numpy as np
import pandas as pd
import networkx as nx



USERS_PATH = "../dataset/preprocessed/filtered_users.csv"
EDGES_PATH = "../dataset/preprocessed/filtered_edges.csv"
#returns the adjacency matrix for the user specified in user_id.
#needs some adjustment to return not only the adjacency matrix but
#the weighted adj_matrix
#returned np.array should be sparse
def get_EgoNetwork(user_id: str, G: nx.DiGraph) -> np.ndarray:
    print(f"Getting Ego Network for user {user_id}...")
    ego = nx.ego_graph(G, user_id, radius=1, undirected=True)
    """
    TODO:
        - weight graph according to mca paper
        - get some statistics, need to read Felix' and Christian's paper to decide which one.
    """
    # toarray() because nx returns a scipy sparse matrix, but giotto (TDA)
    # assumes a np.ndarray
    print("Done!")
    return ego.adjacency_matrix.toarray()
    
if __name__ == "__main__":
    users_df = pd.read_csv(USERS_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    print("Building Graph...")
    G = nx.from_pandas_edgelist(edges_df, "source_id", "target_id", create_using=nx.DiGraph())
    print("Done!")
    #ego = nx.ego_graph(G, "u1595615893", radius=1, undirected=True)
    #Just for testing purposes:
    print(get_EgoNetwork("u1595615893", G))