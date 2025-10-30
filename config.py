

class FilePaths:
    results_path = "results/"
    saved_models_path = "saved_model_files/"
    object_dict_path = "data/object_dicts/"
    dataset_dict_path = "data/dataset_dicts/"
    property_dict_path = "data/property_dicts/"
    dataset_partition_path = "data/dataset_partitions/"


class Constants:
    dataset_name = "synthetic_example"  # "Hague", "delivery3", "bo_em", "gpkg"
    synthetic_folder_name = "example"  # Relevant only if dataset_name is "synthetic"
    evaluation_mode = "matching"  # "blocking", "matching"
    dataset_size_version = 'large'  # 'small', 'medium', 'large'
    matching_cands_generation = 'negative_sampling'  # 'negative_sampling', 'blocking-based'
    neg_samples_num = 2  # 2, 5
    seeds_num = 3
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 1 - train_ratio - val_ratio
    max_ratio_val = 1000  # Avoid infinity values
    load_object_dict = True  # Load existing object dictionary
    save_object_dict = True  # Save the object dictionary
    load_train_items = False  # Load existing preparatory items
    save_property_dict = True  # Save the properties dictionary
    load_property_dict = True  # Load the properties dictionary
    save_dataset_dict = True  # save the dataset dictionary
    load_dataset_dict = False  # load existing dataset dictionary
    file_name_suffix = "grl_291025"  # If set to None, the current exact time will be used
    # contamination_levels = [round(0.05*i, 2) for i in range(1, 11)]  # Contamination levels for contamination models
    contamination_levels = [round(0.05 * i, 2) for i in range(1, 11)]


class TrainingPhase:
    training_ratio = 0.5  # Number of positive samples
    neg_pairs_ratio = 4  # Number of negative samples per positive sample
    run_preparatory_phase = True  # If False, the preparatory phase will not be run


class Features:
    knn_buildings = 0  # Number of nearest buildings to consider
    knn_roads = 0  # Number of nearest roads to consider
    operator = 'division'  # 'division', 'concatenation'
    object_properties = ["bounding_box_width", "bounding_box_length", "area", "perimeter", "perimeter_ind",
                         "volume", "convex_hull_area", "convex_hull_volume", "ave_centroid_distance", "height_diff",
                         "num_floors", "axes_symmetry", "compactness_2d", "compactness_3d", "density",
                         "elongation", "shape_ind", "hemisphericality", "fractality", "cubeness", "circumference",
                         "aligned_bounding_box_width", "aligned_bounding_box_length", "aligned_bounding_box_height",
                         "num_vertices"]

    # object_properties = ["circumference", "density", "convex_hull_area"]
    normalization = 'log_transform'  # 'log_transform', None
    neighborhood = []
    roads = []


class Blocking:
    blocking_method = 'bkafi'  # 'bkafi', 'bkafi_without_SDR', 'ViT-B_32', 'ViT-L_14', 'centroid'
                                # 'coordinates', 'coordinates_transformed'
    cand_pairs_per_item_list = [i for i in range(1, 21)]  # total number of neighbors per each candidate object
    nn_param = cand_pairs_per_item_list[-1] + 1  # number of nearest neighbors to retrieve as candidates
    nbits = 10  # number of bits to use for LSH
    # bkafi_dim_list = [dim for dim in range(1, len(Features.object_properties))] # Number of important features to
    # use for blocking (for the bkafi method)
    bkafi_dim_list = [dim for dim in range(1, len(Features.object_properties))]  # Number of important features to use
    dist_threshold = None  # Define it as a hyperparameter or in a flexible manner
    sdr_factor = False  # If True, the SDR factor will be used in the blocking method
    bkafi_criterion = 'feature_importance'  # 'std', 'feature_importance'


class Models:
    load_trained_models = False
    cv = 3
    model_to_use = 'RandomForestClassifier'
    model_list = ['RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier',
                  'BaggingClassifier', 'XGBClassifier', 'MLPClassifier']
    contamination_model_list = ['RandomForestClassifier', 'BaggingClassifier', 'XGBClassifier']
    blocking_model = 'RandomForestClassifier'  # Used only for blocking and for advanced evaluation
    params_dict = {
                    'RandomForestClassifier': {"n_estimators": [50, 100, 200],
                                               "max_depth": [5, 10],
                                               "min_samples_split": [2],
                                               "max_features": ["sqrt"]},

                    'SVC': {'C': [0.1, 0.5],
                           'kernel': ['rbf'],
                           'gamma': ['scale'],
                           'degree': [2]
                            },

                    'LogisticRegression': {'solver': ['lbfgs', 'saga'],
                                          'multi_class': ['auto'],
                                          'C': [0.01, 0.1, 1]
                                           },

                    'MLPClassifier': {'hidden_layer_sizes': [(64, 32)],
                                      'activation': ['relu'],
                                      'solver': ['adam'],
                                      'batch_size': [16],
                                      'max_iter': [500],
                                      },

                    'AdaBoostClassifier': {'n_estimators': [100, 200],
                                           'learning_rate': [0.1, 0.5, 1.0],
                                           'algorithm': ['SAMME']
                                           },

                    'GradientBoostingClassifier': {'loss': ['log_loss'],
                                                   'learning_rate': [0.01, 0.1],
                                                   'n_estimators': [100, 200],
                                                   'max_depth': [3],
                                                   'min_samples_split': [3],
                                                   'max_features': ['sqrt']
                                                   },

                    'BaggingClassifier': {'n_estimators': [10, 50],
                                          'max_samples': [0.5, 0.8, 1.0],
                                          'max_features': [0.5, 0.8],
                                          'bootstrap': [True, False]
                                          },

                    'XGBClassifier': {'max_depth': [3, 4, 5],
                                      'objective': ['binary:logistic'],
                                      'learning_rate': [0.01, 0.01, 0.1, 0.5],
                                      'n_estimators': [50, 100, 150, 200],
                                      'gamma': [0, 0.1, 1],
                                      }
    }



