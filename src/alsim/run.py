import os
import sys
import argparse
import logging
import pickle
import random
import sklearn
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime

import alsim.paths
import alsim.logutils
import alsim.config
import alsim.data
import alsim.sim
import alsim.classification
import alsim.metrics
import alsim.text


def get_configs(args):
    logger = logging.getLogger("alsim.run.get_configs")
    configs = []

    config_name = args.config_name
    if config_name == "Debug":
        config = alsim.config.DebugConfig()
        configs = [
            config,
        ]
    elif config_name == "Rq1and2":
        config = alsim.config.Rq1and2Config()
        configs = config.get_configs()
    elif config_name == "LcbAl":
        config = alsim.config.LcbAlConfig()
        configs = config.get_configs()
    elif config_name == "LcbAlRej":
        config = alsim.config.LcbAlRejConfig()
        configs = config.get_configs()
    else:
        raise ValueError(f"Unknown config {config_name}.")

    # shuffle the configs, which can help spread out different types of computation captured in the configs
    configs = random.sample(configs, k=len(configs))

    if args.n_runs is not None:
        for config in configs:
            config.n_runs = int(args.n_runs)

    if args.should_print_summary:
        for config in configs:
            config.print_result_summary = args.should_print_summary
    return configs


def log_summary(result):
    logger = logging.getLogger("alsim.run.log_summary")
    logger.info(result.keys())
    logger.info(len(result["combined_metrics_list"]))
    logger.info(result["combined_metrics_list"][-1].keys())

    logger.info("Last-batch difference metrics for each run:")
    for run_metrics in result["run_metrics_list"]:
        last_batch = run_metrics[-1]
        logger.info(last_batch["difference_metrics"])

    logger.info("First-batch true metrics for each run:")
    for run_metrics in result["run_metrics_list"]:
        first_batch = run_metrics[0]
        logger.info(first_batch["true_metrics"])

    logger.info("Last-batch true metrics for each run:")
    for run_metrics in result["run_metrics_list"]:
        last_batch = run_metrics[-1]
        logger.info(last_batch["true_metrics"])

    logger.info("F1 (true & estimated)")
    for i, cm in enumerate(result["combined_metrics_list"]):
        logger.info(
            f"{i:>3}: {cm['true_metrics_f1_mean'] if 'true_metrics_f1_mean' in cm else -1:.4f} {cm['estimated_metrics_f1_mean'] if 'estimated_metrics_f1_mean' in cm else -1:.4f}"
        )
    logger.info(
        f"F1 true (combined, mean and std): {result['combined_metrics_list'][-1]['true_metrics_f1_mean']:.4f} ({result['combined_metrics_list'][-1]['true_metrics_f1_std']:.4f})"
    )
    logger.info(
        f"F1 estimated (combined, mean and std): {result['combined_metrics_list'][-1]['estimated_metrics_f1_mean']:.4f} ({result['combined_metrics_list'][-1]['estimated_metrics_f1_std']:.4f})"
    )


def generate_and_run_batches(working_dir, args, override=False, n_processes=1):
    logger = logging.getLogger("alsim.run.generate_and_run_batches")

    run_name = args.config_name
    output_dir = os.path.join(working_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Will produce outputs in '{output_dir}'.")
    existing_files = os.listdir(output_dir)
    logger.info(f"Found {len(existing_files)} existing files associated with this run.")
    if len(existing_files) > 0:
        for filepath in existing_files:
            os.remove(os.path.join(output_dir, filepath))

    configs = get_configs(args)
    logger.info(f"Generated {len(configs)} configs.")

    n_processes = min(n_processes, len(configs))
    logger.info(f"Will process configs with {n_processes} processes.")
    with mp.Pool(processes=n_processes) as pool:
        pool_results = []
        for i, config in tqdm(enumerate(configs), total=len(configs), desc=run_name):
            config_hash = hash(str(config))
            output_filename = f"result{i}_{config_hash}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            if os.path.exists(output_filepath) and not override:
                logger.info(f"Skipped existing config '{output_filepath}'.")
                if output_filename in existing_files:
                    existing_files.remove(output_filename)
                continue
            result = pool.apply_async(run_batch, (config, output_filepath))
            pool_results.append(result)
            # run_batch(config, output_filepath)
        if len(existing_files) > 0:
            logger.info(
                f"Found {len(existing_files)} files (from a previous run?) that are not reflected in the configs generated for this run. Listing:"
            )
            for filename in existing_files:
                logger.info(filename)
        for i, result in enumerate(pool_results):
            elapsed_time = result.get()
            logger.debug(
                f"Joined config {i} / {len(pool_results)} after {str(elapsed_time)}."
            )
    logger.info(f"Finished processing {len(configs)} configs.")


def run_batch(config, output_filepath):
    logger = logging.getLogger("alsim.run.run_batch")
    config_hash = hash(str(config))
    s = datetime.now()

    training_docs, testing_docs = alsim.data.load_training_and_testing_docs()
    logger.info(
        f"{os.getpid()}: Loaded {len(training_docs)} training and {len(testing_docs)} testing docs."
    )

    # if necessary, load search data
    if config.search_type != "oracle":
        config.search_data = alsim.data.load_search_data()
        if config.search_verbose:
            logger.info(
                f"{os.getpid()}: Loaded search data. ({config.search_data[config.outcome].keys()})"
            )

    # if necessary, load texts and compute vector representations
    if config.vector_source != "roberta_mean_pool":
        alsim.text.add_text_data_to_docs(training_docs, testing_docs)
        alsim.text.precompute_text_vectors(config, training_docs, testing_docs)

    result = alsim.sim.train_and_evaluate_model_config(
        config, training_docs, testing_docs
    )
    # logger.info(f"Finished evaluation of config {config_hash} after {datetime.now() - s}.")

    # save results
    with open(output_filepath, "wb") as outfile:
        pickle.dump(result, outfile)
    logger.info(
        f"Saved result from config {config_hash} to '{output_filepath}' after {datetime.now() - s}."
    )
    if config.print_result_summary:
        log_summary(result)
    return datetime.now() - s


def generate_and_run_max_data_batches(working_dir):
    """
    This function attempts to generate metrics with all of the prospective training data available i.e. labeled in a single batch.
    """
    logger = logging.getLogger("alsim.run.generate_and_run_max_data_batches")
    training_docs, testing_docs = alsim.data.load_training_and_testing_docs()
    logger.info(
        f"Loaded {len(training_docs)} training and {len(testing_docs)} testing docs."
    )

    output_dir = os.path.join(working_dir, "max_data_scores")
    os.makedirs(output_dir, exist_ok=True)

    config = alsim.config.MaxDataConfig()
    configs = config.get_configs()
    configs = random.sample(configs, k=len(configs))

    output_filepath = os.path.join(output_dir, "max_scores.tsv")
    with open(output_filepath, "w") as outfile:
        for config in tqdm(configs, desc="Processing max data configs"):
            s = datetime.now()

            # vectorize X
            X_train = alsim.classification.vectorize_docs(
                config.vector_source, training_docs
            )
            X_test = alsim.classification.vectorize_docs(
                config.vector_source, testing_docs
            )

            # retrieve outcome
            y_train = alsim.sim.get_labels(config.outcome, training_docs)
            y_test = alsim.sim.get_labels(config.outcome, testing_docs)

            if config.use_bbsc and config.undersample_to_proportion is None:
                continue
            pos_pct = np.sum(y_train) / len(y_train)
            if (
                config.use_bbsc
                and pos_pct > config.undersample_to_proportion
                and pos_pct < (1 - config.undersample_to_proportion)
            ):
                # if we won't be downsampling at all, then there's no need to use BBSC
                continue

            true_metrics_list = []
            for i in range(config.n_runs):
                # train the classifier using all available data
                clf, unlabeled_valid_pct_pos = alsim.classification.train_model(
                    X_train, y_train, X_train, config
                )

                # compute the true metrics on the held-out test set
                y_test_pred = clf.predict(X_test)
                y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
                true_metrics = alsim.metrics.compute_metrics(
                    y_test, y_test_pred, y_test_pred_proba
                )
                true_metrics_list.append(true_metrics)

            def get_median_true_metric(metric_key, true_metrics_list):
                return np.median([tm[metric_key] for tm in true_metrics_list])

            f1 = get_median_true_metric("f1", true_metrics_list)
            precision = get_median_true_metric("precision", true_metrics_list)
            recall = get_median_true_metric("recall", true_metrics_list)
            true_pos_pct = get_median_true_metric("true_pos_pct", true_metrics_list)
            pred_pos_pct = get_median_true_metric("pred_pos_pct", true_metrics_list)
            roc_auc = get_median_true_metric("roc_auc", true_metrics_list)
            ap = get_median_true_metric("ap", true_metrics_list)
            accuracy = get_median_true_metric("accuracy", true_metrics_list)

            outfile.write(
                f"{config.outcome}\t{config.use_bbsc}\t{config.undersample_to_proportion}\t{accuracy}\t{f1}\t{precision}\t{recall}\t{roc_auc}\t{ap}\t{true_pos_pct}\t{pred_pos_pct}\n"
            )
            outfile.flush()
            logger.info(
                f"Trained max data '{config.outcome}' (BBSC={config.use_bbsc},u2p={config.undersample_to_proportion}) in {datetime.now() - s}. Acc={accuracy:.4f}; F1={f1:.4f}"
            )
    logger.info("Finished processing max data batches.")


def main():
    alsim.logutils.set_up_logging()
    logger = logging.getLogger("alsim.run.main")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_name", required=False, default="Debug")
    parser.add_argument(
        "--max-data",
        dest="max_data",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument("--n-runs", dest="n_runs", required=False, default=None)
    parser.add_argument("--n-processes", dest="n_processes", required=False, default=16)
    parser.add_argument(
        "--override",
        dest="should_override",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--print-summary",
        dest="should_print_summary",
        required=False,
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    working_dir = os.path.join(alsim.paths.DERIVED_DATA_DIR, "alsim")
    os.makedirs(working_dir, exist_ok=True)
    logger.info(f"Will use working_dir '{working_dir}'.")
    if args.max_data:
        logger.info("Running max_data path")
        generate_and_run_max_data_batches(working_dir)
    else:
        logger.info("Running standard path")
        generate_and_run_batches(
            working_dir,
            args,
            override=args.should_override,
            n_processes=int(args.n_processes),
        )
    logger.info("Finished.")


if __name__ == "__main__":
    main()
