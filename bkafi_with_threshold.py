import config
import argparse
from utils import *
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import RobustScaler
from scipy.spatial import KDTree
import copy
from multiprocessing import Pool, cpu_count
import os


class BKAFIWithThreshold:
    def __init__(self, seed, args):
        self._read_args(seed, args)
        self.bkafi_dim = self._extract_bkafi_dim()
        self.object_dict = self._load_object_dicts()
        self.property_dict = self._load_property_dicts()
        self.idx_mapping_dict = self._get_idx_mapping_dict()
        self.bkafi_dict = {'train': {}, 'test': {}}
        self.percentiles = [round(th, 3) for th in np.arange(0.0, 1.001, 0.01)]
        self.percentile_threhold_dict = self._generate_percentile_thresholds_dict()
        self.nn_dict, self.dists_dict = self._get_retrieved_test_items()
        self.result_dict = self._filter_by_threshold()

    def _read_args(self, seed, args):
        self.seed = seed
        self.dataset_name = args.dataset_name
        self.dataset_size_version = args.dataset_size_version
        self.vector_normalization = args.vector_normalization
        self.sdr_factor = args.sdr_factor
        self.neg_samples_num = args.neg_samples_num
        self.bkafi_criterion = args.bkafi_criterion
        self.model_name = args.model_name
        self.max_k = args.max_k
        self.load_pkls = args.load_pkls
        return

    def _extract_bkafi_dim(self, k_decision=20):
        file_path = self._get_blocking_res_path()
        df = pd.read_csv(file_path, index_col=0)
        df[['bkafi_dim', 'k']] = df.index.to_series().str.split('_', expand=True).astype(int)
        filtered_df = df[df['k'] == k_decision]
        if filtered_df.empty:
            raise ValueError(f"No rows found with k={k_decision}")
        best_row = filtered_df.loc[filtered_df['blocking_recall'].idxmax()]
        return int(best_row['bkafi_dim'])

    def _load_object_dicts(self):
        train_full_path, test_full_path = self._get_object_dict_paths()
        try:
            print(f"Loading train object_dict")
            train_object_dict = joblib.load(train_full_path)
            print(f"Loading test object_dict")
            test_object_dict = joblib.load(test_full_path)
            return {'train': train_object_dict, 'test': test_object_dict}
        except Exception as e:
            print(f"Error loading object_dict: {e}")
            return None

    def _load_property_dicts(self):
        train_full_path = self._get_property_dict_path('train')
        test_full_path = self._get_property_dict_path('test')
        try:
            print(f"Loading train property_dicts")
            train_property_dict = joblib.load(train_full_path)
            print(f"Loading test property_dicts")
            test_property_dict = joblib.load(test_full_path)
            return {'train': train_property_dict, 'test': test_property_dict}
        except Exception as e:
            print(f"Error loading property_dict: {e}")
            return None

    def _get_property_dict_path(self, train_or_test):
        file_name = get_file_name_property_dict()
        property_dict_path = config.FilePaths.property_dict_path
        vector_normalization = 'True' if self.vector_normalization else 'False'
        if not os.path.exists(property_dict_path):
            os.makedirs(property_dict_path)
        property_dict_path = (f"{property_dict_path}{file_name}_{train_or_test}_blocking_"
                              f"{self.dataset_size_version}_neg_samples_num={self.neg_samples_num}_"
                              f"vector_normalization={vector_normalization}_seed={self.seed}.joblib")
        return property_dict_path

    def _get_object_dict_paths(self):
        object_dict_path = f"{config.FilePaths.object_dict_path}{self.dataset_name}/"
        train_full_path = f"{object_dict_path}train_blocking_{self.dataset_size_version}"
        test_full_path = f"{object_dict_path}test_blocking_{self.dataset_size_version}"
        return f"{train_full_path}_seed_{self.seed}.joblib", f"{test_full_path}_seed_{self.seed}.joblib"

    def _get_blocking_output_path(self):
        file_name = get_file_name()
        blocking_results_path = config.FilePaths.results_path + 'blocking_output/'
        vector_normalization = 'True' if self.vector_normalization else 'False'
        sdr_factor = 'True' if self.sdr_factor else 'False'
        if not os.path.exists(blocking_results_path):
            os.makedirs(blocking_results_path)
        blocking_results_path = (f"{blocking_results_path}{file_name}_"
                                 f"{self.dataset_size_version}_neg_samples_num{self.neg_samples_num}"
                                 f"_vector_normalization_{vector_normalization}_sdr_factor_{sdr_factor}_"
                                 f"bkafi_criterion={self.bkafi_criterion}_seed={self.seed}.joblib")
        return blocking_results_path

    def _get_blocking_res_path(self):
        file_name = get_file_name()
        blocking_results_path = config.FilePaths.results_path + 'blocking csv files/'
        vector_normalization = 'True' if self.vector_normalization else 'False'
        sdr_factor = 'True' if self.sdr_factor else 'False'
        if not os.path.exists(blocking_results_path):
            os.makedirs(blocking_results_path)
        blocking_results_path = (f"{blocking_results_path}{file_name}_"
                                 f"{self.dataset_size_version}_neg_samples_num{self.neg_samples_num}"
                                 f"_vector_normalization={vector_normalization}_sdr_factor_{sdr_factor}_"
                                 f"bkafi_criterion={self.bkafi_criterion}.csv")
        return blocking_results_path

    def _get_idx_mapping_dict(self):
        idx_mapping_dict = {'train': {}, 'test': {}}
        for train_or_test in ['train', 'test']:
            for cands_or_index in ['cands', 'index']:
                idx_mapping_dict[train_or_test][cands_or_index] = {ind: orig_ind for ind, orig_ind in
                                                                   enumerate(self.object_dict[train_or_test]
                                                                             [cands_or_index].keys())}
        return idx_mapping_dict

    def _load_pickle(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                return pkl.load(file)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return None

    def _load_data_partition_dict(self):
        dir_path = config.FilePaths.dataset_partition_path
        print(f"Loading dataset_partition_dict from {dir_path}{self.dataset_name}_seed{self.seed}.pkl")
        full_path = f"{dir_path}{self.dataset_name}_seed{seed}.pkl"
        dataset_partition_dict = pkl.load(open(full_path, 'rb'))
        print(f"Loaded dataset_partition_dict: {dataset_partition_dict.keys()}")
        return dataset_partition_dict

    def _load_rel_serch_properties(self):
        file_name = get_file_name()
        models_path = config.FilePaths.saved_models_path
        general_file_name = ''.join((file_name, '_feature_importance_dict'))
        feature_importance_file_name = f'{models_path}{self.dataset_name}/_{general_file_name}_seed={self.seed}.joblib'
        print(f"path: {feature_importance_file_name}")
        try:
            feature_importance_scores = joblib.load(feature_importance_file_name)
            print(f"Loaded feature importance scores from {feature_importance_file_name}")
            rel_props = [feature.split('_ratio')[0] for feature, _ in
                         feature_importance_scores[self.model_name][:self.bkafi_dim]]
            return rel_props
        except Exception as e:
            print(f"Error loading feature importance scores from {feature_importance_file_name}: {e}")
            return None

    def _generate_percentile_thresholds_dict(self):
        matching_pairs_distances = self._get_matching_pairs_distances()
        percentile_thresholds_dict = {percentile: np.percentile(matching_pairs_distances, percentile * 100)
                                      for percentile in self.percentiles}
        return percentile_thresholds_dict

    def _get_matching_pairs_distances(self):
        matching_pairs_dists = self._load_matching_pairs_distances()
        if matching_pairs_dists is not None:
            return matching_pairs_dists
        else:
            return self._compute_train_matching_pairs_dists_wrapper()

    def _load_matching_pairs_distances(self):
        file_path = self._get_blocking_output_path().replace('.joblib', '_matching_pairs_distances.pkl')
        if os.path.exists(file_path) and self.load_pkls is True:
            return self._load_pickle(file_path)
        else:
            return None

    def _get_bkafi_dict(self, rel_search_props, train_or_test):
        local_bkafi_dict = {'cands': {}, 'index': {}}
        for obj_type in ['cands', 'index']:
            for obj_ind in self.object_dict[train_or_test][obj_type].keys():
                local_bkafi_dict[obj_type][obj_ind] = []
                for prop in rel_search_props:
                    prop_val = self.property_dict[train_or_test][prop][obj_type][obj_ind]
                    local_bkafi_dict[obj_type][obj_ind].append(prop_val)
                local_bkafi_dict[obj_type][obj_ind] = np.array(local_bkafi_dict[obj_type][obj_ind])
        return local_bkafi_dict

    def _compute_train_matching_pairs_dists_wrapper(self):
        file_path = self._get_blocking_output_path().replace('.joblib', '_matching_pairs_distances.pkl')
        data_partition_dict = self._load_data_partition_dict()
        training_pairs = data_partition_dict['train']['blocking-based'][self.dataset_size_version][self.neg_samples_num]
        matching_training_pairs_ids = [pair[0] for pair in training_pairs if pair[0] == pair[1]]
        rel_search_props = self._load_rel_serch_properties()
        self.bkafi_dict['train'] = self._get_bkafi_dict(rel_search_props, 'train')
        matching_pairs_dists = self._compute_matching_pair_dists(matching_training_pairs_ids)
        with open(file_path, 'wb') as file:
            pkl.dump(matching_pairs_dists, file)
        print(f"Computed and saved matching pairs distances to {file_path}")
        return matching_pairs_dists

    def _compute_matching_pair_dists(self, matching_training_pairs):
        robust_scaler = RobustScaler()
        cands_dict = self.bkafi_dict['train']['cands']
        index_dict = self.bkafi_dict['train']['index']
        cands_vectors_np = np.array([cands_dict[obj_id] for obj_id in matching_training_pairs], dtype=np.float32)
        index_vectors_np = np.array([index_dict[obj_id] for obj_id in matching_training_pairs], dtype=np.float32)
        cands_vectors_np = robust_scaler.fit_transform(cands_vectors_np)
        index_vectors_np = robust_scaler.transform(index_vectors_np)
        return np.linalg.norm(cands_vectors_np - index_vectors_np, axis=1)

    def _run_kdtree(self, train_or_test):
        robust_scaler = RobustScaler()
        nn_dict, dists_dict = {}, {}
        cands_vectors_np = np.array(list(self.bkafi_dict[train_or_test]['cands'].values()), dtype=np.float32)
        index_vectors_np = np.array(list(self.bkafi_dict[train_or_test]['index'].values()), dtype=np.float32)
        cands_vectors_np = robust_scaler.fit_transform(cands_vectors_np)
        index_vectors_np = robust_scaler.transform(index_vectors_np)
        index = KDTree(index_vectors_np)
        dists, neighbors = index.query(cands_vectors_np, self.max_k)
        for i, _ in enumerate(cands_vectors_np):
            mapped_id = self.idx_mapping_dict[train_or_test]['cands'][i]
            nn_dict[mapped_id] = [self.idx_mapping_dict[train_or_test]['index'][ind] for ind in neighbors[i]]
            dists_dict[mapped_id] = [round(dist, 7) for dist in dists[i]]
        return nn_dict, dists_dict

    def _get_retrieved_test_items(self):
        nn_dict, dists_dict = self._load_test_items()
        if nn_dict is None or dists_dict is None:
            nn_dict, dists_dict = self._run_test_search()
        return nn_dict, dists_dict

    def _run_test_search(self):
        blocking_output_path = self._get_blocking_output_path()
        nn_dict_path = blocking_output_path.replace('.joblib', '_bkafi_test_nn_dict.pkl')
        dists_dict_path = blocking_output_path.replace('.joblib', '_bkafi_test_dists_dict.pkl')
        rel_search_props = self._load_rel_serch_properties()
        self.bkafi_dict['test'] = self._get_bkafi_dict(rel_search_props, 'test')
        nn_dict, dists_dict = self._run_kdtree('test')
        for file_path, dict_data in zip([nn_dict_path, dists_dict_path], [nn_dict, dists_dict]):
            with open(file_path, 'wb') as file:
                pkl.dump(dict_data, file)
            print(f"Saved {file_path}")
        return nn_dict, dists_dict

    def _load_test_items(self):
        nn_dict, dists_dict = None, None
        if not self.load_pkls:
            return nn_dict, dists_dict
        blocking_output_path = self._get_blocking_output_path()
        nn_dict_path = blocking_output_path.replace('.joblib', '_bkafi_test_nn_dict.pkl')
        dists_dict_path = blocking_output_path.replace('.joblib', '_bkafi_test_dists_dict.pkl')
        try:
            nn_dict = self._load_pickle(nn_dict_path)
            dists_dict = self._load_pickle(dists_dict_path)
            if nn_dict is not None and dists_dict is not None:
                print(f"Loaded test items from {nn_dict_path} and {dists_dict_path}")
        except Exception as e:
            print(f"Error loading test items: {e}")
        return nn_dict, dists_dict

    @staticmethod
    def _filter_obj(obj_data):
        obj_id, neighbors, dists, threshold_val = obj_data
        filtered_neighbors = [nid for nid, dist in zip(neighbors, dists) if dist <= threshold_val]
        filtered_dists = [dist for dist in dists if dist <= threshold_val]
        return obj_id, filtered_neighbors, filtered_dists

    def _filter_by_threshold(self):
        filtered_nn_dict = copy.deepcopy(self.nn_dict)
        filtered_dists_dict = copy.deepcopy(self.dists_dict)
        res_dict = {}
        percentiles_reverse = sorted(self.percentile_threhold_dict.keys(), reverse=True)
        for percentile in tqdm.tqdm(percentiles_reverse):
            threshold_val = self.percentile_threhold_dict[percentile]
            obj_inputs = [
                (obj_id, filtered_nn_dict[obj_id], filtered_dists_dict[obj_id], threshold_val)
                for obj_id in filtered_nn_dict
            ]
            with Pool(cpu_count()-2) as pool:
                results = pool.map(BKAFIWithThreshold._filter_obj, obj_inputs)
            filtered_nn_dict = {obj_id: neighbors for obj_id, neighbors, _ in results}
            filtered_dists_dict = {obj_id: dists for obj_id, _, dists in results}
            res_dict[percentile] = self._compute_stats(filtered_nn_dict, threshold_val)
        return res_dict

    def _compute_stats(self, nn_dict, threshold_val):
        cand_ids = set(self.object_dict['test']['cands'].keys())
        index_ids = set(self.object_dict['test']['index'].keys())
        intersection = cand_ids.intersection(index_ids)
        recall = round(sum(1 for obj_id in intersection if obj_id in nn_dict and obj_id in nn_dict[obj_id]) /
                       len(intersection), 3)
        cand_pairs_num = sum([len(neighbors) for neighbors in nn_dict.values()])
        reduction_ratio = round(1 - cand_pairs_num / (len(cand_ids) * len(index_ids)), 8)
        return {'recall': recall,
                'cand_pairs_num': cand_pairs_num,
                'reduction_ratio': reduction_ratio,
                'threshold_val': threshold_val
                }


def generate_final_result_csv_BKAFI_with_threshold(result_dict, args):
    aggregated_res = {}
    for seed, seed_results in result_dict.items():
        for percentile, metric_dict in seed_results.items():
            if percentile not in aggregated_res:
                aggregated_res[percentile] = []
            aggregated_res[percentile].append(metric_dict)

    averaged_results = []
    for percentile, metrics_list in aggregated_res.items():
        avg_metrics = {
            'percentile': percentile,
            'recall': round(100 * np.mean([m['recall'] for m in metrics_list]), 2),
            'cand_pairs_num': round(np.mean([m['cand_pairs_num'] for m in metrics_list])),
            'reduction_ratio': round(100 * np.mean([m['reduction_ratio'] for m in metrics_list]), 6),
            'threshold_val': round(np.mean([m['threshold_val'] for m in metrics_list]), 3)
        }
        averaged_results.append(avg_metrics)

    df = pd.DataFrame(averaged_results).sort_values(by='percentile')
    results_path = config.FilePaths.results_path + f"blocking csv files/"
    file_name = get_file_name('bkafi')
    res_file_path = f"{results_path}{file_name}_{args.dataset_size_version}_BKAFI_with_thresholds.csv"
    df.to_csv(res_file_path, index=False)
    print(f"Saved final results to {res_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--seeds_num', type=int, default=config.Constants.seeds_num)
    parser.add_argument('--dataset_size_version', type=str, default=config.Constants.dataset_size_version)
    parser.add_argument('--vector_normalization', type=str2bool, default=True)
    parser.add_argument('--sdr_factor', type=str2bool, default=False)
    parser.add_argument('--neg_samples_num', type=int, default=config.Constants.neg_samples_num)
    parser.add_argument('--bkafi_criterion', type=str, default=config.Blocking.bkafi_criterion)
    parser.add_argument('--max_k', type=int, default=20)
    parser.add_argument('--model_name', type=str, default='RandomForestClassifier')
    parser.add_argument('--load_pkls', type=str2bool, default=True)

    args = parser.parse_args()
    result_dict = {}
    for seed in range(1, args.seeds_num+1):
        print(f"Seed: {seed}")
        print(3*'--------------------------')
        bkafi_with_threshold_manager = BKAFIWithThreshold(seed, args)
        result_dict[seed] = bkafi_with_threshold_manager.result_dict
    generate_final_result_csv_BKAFI_with_threshold(result_dict, args)
    print("Done!")
