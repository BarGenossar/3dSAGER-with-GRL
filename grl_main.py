import argparse
import warnings
import config
from utils import define_logger, print_config, generate_final_result_csv
from grl_pipeline import GRLPipelineManager
from utils import str2bool

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--suffix', type=str, default='291025')
    parser.add_argument('--load_cached_graphs',  type=str2bool, default=True)
    parser.add_argument('--load_cached_pairs', type=str2bool, default=True)
    parser.add_argument('--min_surfaces_num', type=int, default=20)
    parser.add_argument('--neg_pairs_num', type=int, default=2)
    parser.add_argument('--training_epochs', type=int, default=3)
    parser.add_argument('--pair_aggregation', type=str, default='abs_diff')  # 'concat', 'abs_diff', 'division' or 'all'
    parser.add_argument('--evaluation_mode', type=str, default='matching')
    # ADD lr, weight_decay, gnn_layers_num, batch_size, hidden_dim1, hidden_dim2, hidden_dim3, dropout_rate,
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--gnn_layers_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim1', type=int, default=64)
    parser.add_argument('--hidden_dim2', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.2)






    args = parser.parse_args()

    logger = define_logger(args)
    # print_config(logger, args)

    result_dict = {}
    for seed in range(1, args.seeds_num + 1):
        logger.info(f"Seed: {seed}")
        logger.info(3 * '--------------------------')

        pipeline_manager = GRLPipelineManager(seed, logger, args)
        result_dict[seed] = pipeline_manager.eval_artifacts['metrics']

    # placeholder: we only want to check data generation right now
    # but if you already have generate_final_result_csv in utils, keep it
    generate_final_result_csv(result_dict, args)

    logger.info("Done!")
