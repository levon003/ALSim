from copy import copy, deepcopy

ALL_OUTCOMES = [
    "is_vb",
    "is_overall_5",
    "is_cat_Books",
    "is_cat_Movie/TV",
]


def create_configs_for_outcomes(root_config, outcomes=ALL_OUTCOMES):
    configs = []
    for outcome in outcomes:
        config = deepcopy(root_config)
        config.outcome = outcome
        configs.append(config)
    return configs


class BinaryTextClassifierConfig:
    def __init__(self):
        self.n_runs = 5
        self.batch_size = 10
        self.n_batches = 200 // self.batch_size
        self.outcome = "is_vb"

        self.vector_source = "roberta_mean_pool"
        self.vector_max_features = None
        self.vector_ngram_range = (1, 1)  # unigrams by default

        self.should_tqdm_runs = False
        self.print_result_summary = False

        # learner configuration
        self.learner = "logreg"
        self.use_bbsc = False
        self.undersample_to_proportion = 0.2

        self.metric_strategy = "cv"
        self.estimate_metrics_after_batch = 0
        self.estimate_metrics_batch_allowlist = None  # [4, 9, 14, 19]

        # sampling strategy and configuration
        self.sample_strategy = "all_random"
        self.kmeans_k = 100  # 100 is the number used in the surprisal embeddings paper
        self.n_reserved_random = 0

        self.pool_strategy = "all"
        self.search_data = {}
        self.search_type = "oracle"
        self.search_ranking_strategy = "default"
        self.search_pool_type = "bm25"
        # search_type in ['oracle', 'unigram', 'bigram', 'human']
        # search_ranking_strategy in ['default', 'bm25', 'rm3']
        # search_pool_type in ['bkm', 'bm25'] ; 'rm3' is no longer supported (in part due to its incombatilibility with the bigram AND queries), and bkm results are not reported
        # note: bkm = 'boolean keyword matching'
        self.oracle_proportion_positive = 0.5
        self.search_verbose = False

        # LCB-AL parameters
        self.lcbal_verbose = False  # determine whether to do additional logging about the state of the LCB-AL process, which is useful for debugging
        self.use_lcbal_classifier = True
        self.lcbal_use_rejection_sampling = False
        self.lcbal_p_scale = 2
        self.lcbal_C_scale = 10000

    def update_from_dict(self, new_vals, check_for_existing_key=True):
        for key, val in new_vals.items():
            assert type(key) == str
            if check_for_existing_key:
                assert key in self.__dict__, key
            self.__dict__[key] = val

    def __str__(self):
        return str(self.__dict__)

    def get_configs(self, config_value_map):
        inds = [0 for i in range(len(config_value_map))]
        config_key_list = []
        value_lists = []
        for config_key, value_list in config_value_map.items():
            config_key_list.append(config_key)
            value_lists.append(value_list)
        if (
            len(config_key_list) == 0
        ):  # special case: empty config_value_map should just produce this config itself
            return [
                deepcopy(self),
            ]
        configs = []
        finished_processing = False
        while not finished_processing:
            config = deepcopy(self)
            config.update_from_dict(
                {
                    config_key: value_list[ind]
                    for config_key, value_list, ind in zip(
                        config_key_list, value_lists, inds
                    )
                }
            )
            configs.append(config)
            # update the inds
            i = len(inds) - 1
            while i >= 0:
                inds[i] += 1
                if inds[i] == len(value_lists[i]):
                    if i == 0:
                        finished_processing = True
                    # need to carry the 1
                    inds[i] = 0
                    i -= 1
                    if i == -1:
                        break
                else:
                    i = -1  # done updating inds
        return configs


class DebugConfig(BinaryTextClassifierConfig):
    def __init__(self):
        super().__init__()
        self.should_tqdm_runs = True
        self.print_result_summary = True

        # override any defaults for testing here
        self.sample_strategy = "all_random"

        self.search_verbose = False
        # self.pool_strategy = 'search_until_model'
        # self.search_type = 'human'
        # self.search_ranking_strategy = 'bm25'

        self.n_runs = 3

        # self.lcbal_verbose = True
        # self.use_lcbal_classifier = False
        # self.lcbal_use_rejection_sampling = True
        # self.lcbal_p_scale = 10
        # self.lcbal_C_scale = 10000

        self.vector_source = "tfidf"
        self.vector_max_features = 50000
        self.vector_ngram_range = (1, 2)

        self.batch_size = 100
        self.n_batches = 2


class Rq1and2Config(BinaryTextClassifierConfig):
    def __init__(self):
        super().__init__()
        self.n_runs = 100

    def get_configs(self):
        config_value_map = {
            "outcome": ALL_OUTCOMES,
            "sample_strategy": ["all_uncertainty", "all_random", "all_kmeans"],
            "metric_strategy": ["cv", "random_only_cv"],
            "n_reserved_random": [0, 2],
        }
        all_configs = super().get_configs(config_value_map)
        configs = []
        for config in all_configs:
            if (
                config.metric_strategy == "random_only_cv"
                and config.n_reserved_random == 0
            ):
                continue
            if (
                config.sample_strategy == "all_random"
                and config.metric_strategy == "random_only_cv"
            ):
                continue
            if config.sample_strategy == "all_random" and config.n_reserved_random == 2:
                continue
            configs.append(config)
        return configs


class LcbAlConfig(BinaryTextClassifierConfig):
    def __init__(self):
        super().__init__()
        self.n_runs = 100
        self.sample_strategy = "lcbal"

    def get_configs(self):
        config_value_map = {
            "outcome": ALL_OUTCOMES,
            "use_lcbal_classifier": [False, True],
            "lcbal_p_scale": [1, 2, 5, 10, 100],
            "lcbal_C_scale": [1000, 10000, 100000],
        }
        all_configs = super().get_configs(config_value_map)
        configs = []
        for config in all_configs:
            if (
                config.metric_strategy == "random_only_cv"
                and config.n_reserved_random == 0
            ):
                continue
            configs.append(config)
        return configs


class LcbAlRejConfig(BinaryTextClassifierConfig):
    """
    Config for LCB-AL with comparisons for Rejection Sampling
    """

    def __init__(self):
        super().__init__()
        self.n_runs = 100
        self.sample_strategy = "lcbal"
        self.lcbal_p_scale = 4

    def get_configs(self):
        config_value_map = {
            "outcome": ALL_OUTCOMES,
            "use_lcbal_classifier": [False, True],
            "lcbal_use_rejection_sampling": [False, True],
        }
        all_configs = super().get_configs(config_value_map)
        configs = []
        for config in all_configs:
            if config.use_lcbal_classifier and config.lcbal_use_rejection_sampling:
                # these settings conflict with each other, since both change how true performance is computed
                continue
            configs.append(config)
        return configs
