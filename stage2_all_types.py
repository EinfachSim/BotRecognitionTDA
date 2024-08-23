import gtda as tda
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import Pipeline
from gtda.diagrams import Scaler, PersistenceImage, PersistenceLandscape
import math
import numpy as np

"""
This method computed the persistence diagrams obtained by WeightedVietorisRips
filtration (and maybe others).
takes in one ego network as np.ndarray (adjacency/distance matrix)
returns a list of np.ndarrays of shape (1,n_features, 3)
each entry in that list corresponds to a persistence diagram in the given homology
dimension
"""
def get_persistence_diagram(ego: np.ndarray, pipeline: tda.pipeline.Pipeline) -> np.ndarray:
    #VR needs to have np.inf for absent edges, NOT ZERO!
    ego[ego == 0] = np.inf
    
    diagram = pipeline.fit_transform([ego])
    PI = PersistenceImage(sigma=1)
    p_image = PI.fit_transform(diagram)
    p_landscape = PersistenceLandscape().fit_transform(diagram)
    
    # PI.plot(p_image, homology_dimension_idx=1).show()
    # p_image_r = p_image[0, 0, :, :]
    # p_image_g = p_image[0, 1, :, :]
    # p_image_b = p_image[0, 2, :, :]
    # rgb_uint8 = np.dstack((p_image_r,p_image_g,p_image_b))
    # plt.imshow(rgb_uint8)
    # plt.show()
    return (diagram, p_image, p_landscape)

if __name__ == "__main__":
    steps = [
        # n_jobs set to -1 leads to using all processors
        ("VR", VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0,1,2], n_jobs=-1, max_edge_length=100))
        #("Scaling", Scaler(function=lambda x: np.max(x)))
        #Need to read more for these two
        #("Persistence Images", PersistenceImage()),
        #("Persistence Landscape", PersistenceLandscape())
    ]
    pipeline = Pipeline(steps)
    from os import listdir
    from os.path import isfile, join
    PATH = "../data_from_pod/"
    k = 1
    node_files = [f for f in listdir(PATH + "ego_networks_copy") if isfile(join(PATH + "ego_networks_copy", f))]
    for file in node_files:
        node_id = file.split("_")[-1]
        print(f"CURRENTLY AT FILE {k} OF {len(node_files)} WITH FILE NAME {file}")
        ego_nw = np.load(PATH + f"ego_networks_copy/{file}")
        diag, pim, pla = get_persistence_diagram(ego_nw, pipeline)
        np.save(PATH + f"p_diagrams/p_diag_{node_id}", diag)
        np.save(PATH + f"p_images/p_im_{node_id}", pim)
        np.save(PATH + f"p_landscapes/p_land_{node_id}", pla)
        k += 1