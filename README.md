# 3D Entity Resolution with Graph Representation Learning

This repository implements a graph-based framework for matching 3D geospatial objects. The system takes CityJSON building models, converts them into semantic hierarchical graphs (building → surfaces → openings), extracts geometric and semantic features, and trains a Graph Neural Network (GNN) to determine whether two 3D objects represent the same real-world entity.

Two objects are then compared through a pair-aggregation mechanism and classified as a match (1) or non-match (0).  

---

## Visual Summary

### Semantic Decomposition of a CityJSON Building

A CityJSON building is decomposed into its structural elements:

- Main Object → Ground, Wall, Roof surfaces  
- Surfaces → Windows and Doors  
- Geometry → Polygon meshes with coordinates

These elements form the nodes of the graph, with edges representing semantic and hierarchical relations.


---
### Input Data Format
The system expects a pre-generated dictionary named data_partition_dict that contains the train and test splits. Each split may include various sampling strategies such as negative sampling or blocking-based sampling.

A typical structure looks like:

```
{
    'train': {
        'negative_sampling': {...},
        'blocking-based': {
            'small': {...},
            'large': {
                2: [
                    ('uuidA1', 'uuidB1'),
                    ('uuidA2', 'uuidB2'),
                    ...
                ],
                5: [
                    ('uuidA3', 'uuidB3'),
                    ...
                ]
            }
        }
    },
    'test': {
        ...
    }
}
```

Pairs are represented as tuples of object UUIDs. The code resolves each UUID into its corresponding CityJSON building structure and constructs the graph internally.

---
### Running the Code
```
python grl_main.py
```
This loads the configuration from config.py, reads the dataset partitions, builds the graphs, trains the matcher and saves results to disk.

You can launch automated experiments covering multiple parameter combinations with:

```
bash run_grl_experiments.sh
```

After experiments complete, extract the best-performing setup (based on F1 score):

```
python find_best_params.py
```
