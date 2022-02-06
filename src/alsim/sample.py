import numpy as np
import sklearn
import logging
from sklearn.cluster import KMeans

from alsim.classification import vectorize_docs
from alsim.data import get_labels
import alsim.lcbal

MIN_SEARCH_RETURN_THRESHOLD = 5


def get_batch_sample(
    config,
    batch_size,
    curr_batch,
    n_batches,
    selected_training_docs,
    training_pool_docs,
    prev_batch_metrics,
    existing_classifier,
):
    """
    curr_batch is zero-indexed, so the first batch is curr_batch
    """
    logger = logging.getLogger("alsim.sample.get_batch_sample")
    name = config.sample_strategy
    n_reserved_random = config.n_reserved_random
    pool_strategy = config.pool_strategy
    search_type = config.search_type
    search_ranking_strategy = config.search_ranking_strategy
    search_pool_type = config.search_pool_type

    prev_n_train = (
        prev_batch_metrics["n_train"] if "n_train" in prev_batch_metrics else 0
    )
    prev_n_train_pos = (
        prev_batch_metrics["n_train_pos"] if "n_train_pos" in prev_batch_metrics else 0
    )
    pct_pos = prev_n_train_pos / prev_n_train if prev_n_train > 0 else 0.0

    should_filter = False
    pool_docs = training_pool_docs
    if pool_strategy == "all":
        pool_docs = training_pool_docs
    elif pool_strategy == "search_until_model":
        if existing_classifier is None:
            should_filter = True
            if curr_batch > 0 and pct_pos >= 0.9:
                # we are lacking negative documents, so actually maybe let's not search....
                # basically, this condition forces a round of default when using narrow search queries
                should_filter = False
    elif pool_strategy == "search_until_10pct":
        if pct_pos <= 0.1:
            should_filter = True
    elif pool_strategy == "search_until_25pct":
        if pct_pos <= 0.25:
            should_filter = True
    elif pool_strategy == "search_with_p20":
        # search with probability 20
        if np.random.random() <= 0.2:
            should_filter = True
    elif pool_strategy == "search_always":
        should_filter = True

    # TODO given that the search_terms are associated with the configuration,
    # we could easily make this faster by generating a document_id index
    # ahead of time. Then just quickly build index of index to document_id.
    if should_filter:
        if search_type == "oracle":
            # oracle sampling retrieves a random sample at a specific class proportion
            # by default, it returns half positive and half negative documents
            proportion_pos = config.oracle_proportion_positive
            y = get_labels(config.outcome, training_pool_docs)
            pos_to_sample_count = int(np.ceil(batch_size * proportion_pos))
            neg_to_sample_count = batch_size - pos_to_sample_count

            pos_inds = np.nonzero(y == 1)[0]
            pos_inds_to_sample = np.random.choice(
                pos_inds, size=pos_to_sample_count, replace=False
            )

            neg_inds = np.nonzero(y == 0)[0]
            neg_inds_to_sample = np.random.choice(
                neg_inds, size=neg_to_sample_count, replace=False
            )
            pos_docs = [training_pool_docs[pos_ind] for pos_ind in pos_inds_to_sample]
            neg_docs = [training_pool_docs[neg_ind] for neg_ind in neg_inds_to_sample]
            pool_docs = pos_docs + neg_docs
            assert (
                len(pool_docs) == batch_size
            ), f"Oracle retrieved more than the batch size ({len(pool_docs)} > {batch_size})."
        elif (
            search_type == "unigram"
            or search_type == "bigram"
            or search_type == "human"
        ):
            search_data = config.search_data[config.outcome]

            if search_type == "unigram":
                sd = search_data["broad_results"]
            elif search_type == "bigram":
                sd = search_data["narrow_results"]
            elif search_type == "human":
                sd = search_data["human_results"]
            else:
                raise ValueError(f"Unknown search type {search_type}.")
            # sd is a dict of search_query -> result_data

            available_document_ids = set(
                [doc["document_id"] for doc in training_pool_docs]
            )
            search_query_options = np.array(list(sd.keys()))
            # shuffle the available query strings
            np.random.shuffle(search_query_options)
            identified_valid_search = False
            i = 0
            while not identified_valid_search and i < len(search_query_options):
                search_query = search_query_options[i]
                i += 1
                result_data = sd[search_query]
                # assert search_pool_type in result_data
                search_result_dict = result_data[search_pool_type]
                document_id_list = search_result_dict["document_id_list"]
                searched_document_ids = set(document_id_list)
                available_and_searched = searched_document_ids & available_document_ids
                if len(available_and_searched) < MIN_SEARCH_RETURN_THRESHOLD:
                    # reject a search if it would return fewer than 5 documents
                    # in other words, look at the next search_query
                    continue

                pool_docs = [
                    doc
                    for doc in training_pool_docs
                    if doc["document_id"] in available_and_searched
                ]
                assert len(pool_docs) >= MIN_SEARCH_RETURN_THRESHOLD
                identified_valid_search = True
            if i == len(search_query_options) and not identified_valid_search:
                logger.warning(
                    f"Exausted all search_query_options without finding a match. Bailing on search. {search_query_options}"
                )
                should_filter = False
            if config.search_verbose:
                logger.info(
                    f"Search query: '{search_query}'; {len(available_and_searched)} available docs, {len(pool_docs)} pool docs. ({len(search_query_options)} queries available, rejected {i-1} queries.)"
                )
        else:
            raise ValueError(f"Unknown search type {search_type}.")

    if name == "all_random":
        sample_type = "random"
    elif name == "all_kmeans":
        sample_type = "kmeans"
    elif name == "all_uncertainty":
        sample_type = "uncertainty"
    elif name == "cyclical":
        if curr_batch % 3 == 0:
            sample_type = "random"
        elif curr_batch % 3 == 1:
            sample_type = "kmeans"
        elif curr_batch % 3 == 2:
            sample_type = "uncertainty"
    elif name == "diversity_first":
        sample_type = "kmeans" if curr_batch <= 10 else "uncertainty"
    elif name == "lcbal":
        sample_type = "lcbal"

    if sample_type == "kmeans" and pool_docs != training_pool_docs:
        # don't use global kmeans configuration when we are filtering the training pool
        sample_type = "new_kmeans"

    n_to_sample = batch_size
    pre_selected_inds = None
    if n_reserved_random > 0 and name != "random":
        # need to sample randomly
        # note: this sample ignores the pool_strategy...
        n_reserved_random = min(n_reserved_random, batch_size)
        pre_selected_inds = list(
            np.random.choice(range(len(training_pool_docs)), size=2, replace=False)
        )
        n_to_sample -= 2
        assert n_to_sample > 0

    if existing_classifier is None and sample_type in [
        "uncertainty",
        "most_likely_positive",
    ]:
        # classifier required for this sample type but it does not exist yet
        # so use random sampling instead
        sample_type = "random"

    if should_filter and len(pool_docs) < len(training_pool_docs):
        if search_ranking_strategy != "default":
            # we need to apply an alternative ranking strategy
            sample_type = search_ranking_strategy
        if sample_type == "lcbal":
            # LCB-AL is poorly defined with a changing data pool
            # for now, just don't use LCB-AL when the pool is filtered
            sample_type = "random"

    # given the sample_type, select documents from pool_docs
    # each block in the if statement below should set selected_inds
    # to be a list of indices in pool_docs
    n_to_sample = min(n_to_sample, len(pool_docs))
    if n_to_sample == len(pool_docs):
        selected_inds = list(range(n_to_sample))
    elif sample_type == "random":
        selected_inds = list(
            np.random.choice(range(len(pool_docs)), size=n_to_sample, replace=False)
        )
        assert len(selected_inds) == n_to_sample
    elif sample_type == "uncertainty":
        # predict on the pool docs using the classifier
        X_pool = vectorize_docs(config.vector_source, pool_docs)
        y_pool_pred_proba = existing_classifier.predict_proba(X_pool)[:, 1]

        y_pred_abs = np.abs(y_pool_pred_proba - 0.5)
        sort_inds = np.argsort(y_pred_abs)
        # select the batch_size docs with the highest uncertainty
        selected_inds = list(sort_inds[:n_to_sample])
        assert len(selected_inds) == n_to_sample
    elif sample_type == "most_likely_positive":
        # predict on the pool docs using the classifier
        X_pool = vectorize_docs(config.vector_source, pool_docs)
        y_pool_pred_proba = existing_classifier.predict_proba(X_pool)[:, 1]

        # sort by the highest probability and select top n docs
        sort_inds = np.argsort(y_pool_pred_proba)
        selected_inds = list(sort_inds[:n_to_sample])
        assert len(selected_inds) == n_to_sample
    elif sample_type == "kmeans":
        assert len(pool_docs) == len(
            training_pool_docs
        ), f"{len(pool_docs)}, {len(training_pool_docs)}"
        should_run_kmeans = False
        if "is_kmeans_sample_eligible" not in pool_docs[0]:
            should_run_kmeans = True
        else:
            # assert np.all(['is_kmeans_sample_eligible' in doc for doc in pool_docs]), f"{np.sum(['is_kmeans_sample_eligible' in doc for doc in pool_docs])} / {len(pool_docs)}"
            kmeans_pool_docs = [
                doc for doc in pool_docs if doc["is_kmeans_sample_eligible"]
            ]
            if len(kmeans_pool_docs) == 0:
                # we burned through all samples in the pool, so run it again
                should_run_kmeans = True
        if should_run_kmeans:
            # run kmeans and set pool_docs with cluster info
            X_pool = vectorize_docs(config.vector_source, pool_docs)
            X_pool_normed = sklearn.preprocessing.normalize(X_pool, norm="l2")
            kmeans = KMeans(
                n_clusters=config.kmeans_k,
                max_iter=10,
            )
            cluster_distances = kmeans.fit_transform(X_pool_normed)
            # cluster_assignments = kmeans.predict(X_pool_normed)

            # identify the docs that are closest to each of the centroids
            closest_instance_inds = np.argmin(cluster_distances, axis=0)
            kmeans_selected_inds = list(closest_instance_inds)
            for i, doc in enumerate(pool_docs):
                is_kmeans_sample_eligible = i in kmeans_selected_inds
                doc["is_kmeans_sample_eligible"] = is_kmeans_sample_eligible
            kmeans_pool_docs = [
                doc for doc in pool_docs if doc["is_kmeans_sample_eligible"]
            ]
            # done with kmeans, eligible docs placed in kmeans_pool_docs

        # sample randomly from the kmeans_pool_docs
        kmeans_selected_inds = list(
            np.random.choice(
                range(len(kmeans_pool_docs)),
                size=min(n_to_sample, len(kmeans_pool_docs)),
                replace=False,
            )
        )
        # convert inds in kmeans subset to full pool
        selected_inds = [
            pool_docs.index(kmeans_pool_docs[i]) for i in kmeans_selected_inds
        ]
        # TODO in the case where insufficient cluster centers are available, should probably do something reasonable to pad the batch (such as choosing randomly, or rerunning k-means)
        assert len(selected_inds) > 0
    elif sample_type == "new_kmeans":
        X_pool = vectorize_docs(config.vector_source, pool_docs)
        X_pool_normed = sklearn.preprocessing.normalize(X_pool, norm="l2")
        kmeans = KMeans(
            n_clusters=n_to_sample,
            max_iter=10,
        )
        cluster_distances = kmeans.fit_transform(X_pool_normed)
        # cluster_assignments = kmeans.predict(X_pool_normed)

        # identify the docs that are closest to each of the centroids
        closest_instance_inds = np.argmin(cluster_distances, axis=0)
        selected_inds = list(closest_instance_inds)
    elif sample_type == "bm25" or sample_type == "rm3":
        # use the bm25 scores on the search_query to rank the docs
        result_data = sd[search_query]
        search_result_dict = result_data[sample_type]
        document_id_score_dict = search_result_dict["document_id_score_dict"]
        scores = [
            document_id_score_dict[doc["document_id"]]
            if doc["document_id"] in document_id_score_dict
            else 0.0
            for doc in pool_docs
        ]
        ordered_doc_inds = np.argsort(scores)
        # finally, we select the documents with the highest scores
        selected_inds = list(ordered_doc_inds[-n_to_sample:])
    elif sample_type == "lcbal":
        selected_inds = alsim.lcbal.run_lcbal(
            config, curr_batch, n_to_sample, pool_docs
        )
    else:
        raise ValueError(f"Unknown sample type {sample_type}.")

    # translate indices wrt pool_docts to indices wrt training_pool_docs
    # if filtering was done on the training_pool_docs
    if should_filter:
        # we need to map the selected indices to the full-list indices
        full_pool_selected_inds = []
        for ind in selected_inds:
            doc = pool_docs[ind]
            new_ind = training_pool_docs.index(doc)
            full_pool_selected_inds.append(new_ind)
        selected_inds = full_pool_selected_inds

        # we need to override the sample_type
        # basically, this stops us from considering filtered results
        # to be randomly sampled (important for random_only_cv)
        if should_filter:
            sample_type += "_search"

    # merge in the pre_selected_inds and record the per-doc sample type
    if pre_selected_inds is not None:
        # remove any duplicate inds
        for pre_selected_ind in pre_selected_inds:
            if pre_selected_ind in selected_inds:
                selected_inds.remove(pre_selected_ind)
        # then, combine pre-selected and selected indices
        # selected_inds = pre_selected_inds + selected_inds
        pre_selected_inds_tup = [
            (selected_ind, "random") for selected_ind in pre_selected_inds
        ]
        selected_inds_tup = pre_selected_inds_tup + [
            (selected_ind, sample_type) for selected_ind in selected_inds
        ]
    else:
        selected_inds_tup = [
            (selected_ind, sample_type) for selected_ind in selected_inds
        ]

    return selected_inds_tup
