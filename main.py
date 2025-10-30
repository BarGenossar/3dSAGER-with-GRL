import config
import argparse
from utils import *
from pipelines import PipelineManager
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--evaluation_mode', type=str, default=config.Constants.evaluation_mode)
    parser.add_argument('--run_preparatory_phase', type=bool, default=config.TrainingPhase.run_preparatory_phase)
    parser.add_argument('--blocking_method', type=str, default=config.Blocking.blocking_method)
    parser.add_argument('--seeds_num', type=int, default=config.Constants.seeds_num)
    parser.add_argument('--dataset_size_version', type=str, default=config.Constants.dataset_size_version)
    parser.add_argument('--vector_normalization', type=str2bool, default=True)
    parser.add_argument('--sdr_factor', type=str2bool, default=False)
    parser.add_argument('--neg_samples_num', type=int, default=config.Constants.neg_samples_num)
    parser.add_argument('--bkafi_criterion', type=str, default=config.Blocking.bkafi_criterion)
    parser.add_argument('--run_blocker_train', type=str2bool, default=False)
    parser.add_argument('--matching_cands_generation', type=str,
                        default=config.Constants.matching_cands_generation)
    parser.add_argument('--contamination_mode', type=str2bool, default=False)


    args = parser.parse_args()
    logger = define_logger()
    print_config(logger, args)
    result_dict = {}
    for seed in range(1, args.seeds_num+1):
        logger.info(f"Seed: {seed}")
        logger.info(3*'--------------------------')
        pipeline_manager_obj = PipelineManager(seed, logger, args)
        result_dict[seed] = pipeline_manager_obj.result_dict
    if not args.run_blocker_train:
        generate_final_result_csv(result_dict, args)
    logger.info("Done!")


