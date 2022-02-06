
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random
import sklearn
import logging

import alsim.sample
import alsim.classification
import alsim.metrics
import alsim.utils

from alsim.classification import vectorize_docs
from alsim.data import get_labels

def train_and_evaluate_model(curr_batch, config, selected_training_docs, validation_docs, testing_docs):
    learner, metric_strategy = config.learner, config.metric_strategy
    assert learner == 'logreg'
    
    # vectorize X
    X_train = vectorize_docs(config.vector_source, selected_training_docs)
    X_test = vectorize_docs(config.vector_source, testing_docs)
    X_unlabeled_validation = vectorize_docs(config.vector_source, validation_docs)

    # retrieve outcome
    y_train = get_labels(config.outcome, selected_training_docs)
    y_test = get_labels(config.outcome, testing_docs)
    #y_unlabeled_validation = get_labels(config.outcome, validation_docs)
    
    # decide if a classifier can be trained
    # (needs 2+ pos and 2+ neg samples in labeled pool)
    should_train_classifier = False
    pos_count = np.sum(y_train)
    if pos_count > 1 and pos_count < len(selected_training_docs) - 1:
        should_train_classifier = True
    # if not, return an empty metrics object and an empty classifier
    if not should_train_classifier:
        model_metrics = {
                    'n_train': len(y_train),
                    'n_train_pos': np.sum(y_train),
                    'true_metrics': {},
                    'estimated_metrics': {},
                    'difference_metrics': {}
                }
        return model_metrics, None
    
    # train the global classifier using all available data
    clf, unlabeled_valid_pct_pos = alsim.classification.train_model(X_train, y_train, X_unlabeled_validation, config)
    new_classifier = clf
    unlabeled_valid_pct_pos_list = [unlabeled_valid_pct_pos,]
    # note: when using lcbal, the positive class proportion predictions are invalid; see these TODOs:
    # TODO if lcbal, use cached classifier in the config as the new_classifier, use that classifier to compute unlabeled_valid_pct_pos
    # TODO alternately, could basically just ignore the cached classifier? use config.lcbal_clf for sampling only, but train a new classifier for performance evaluation.  Could be a good stop-gap measure during testing, with outs to compare the trained model to LCBAL's inferred model
    if config.sample_strategy == 'lcbal' and config.use_lcbal_classifier and 'lcbal_params' in config.__dict__:
        clf = config.lcbal_params.clf
    elif config.sample_strategy == 'lcbal' and config.lcbal_use_rejection_sampling and 'lcbal_params' in config.__dict__:
        logger = logging.getLogger("alsim.sim.lcbal.use_rejection_sampling")
        inds = config.lcbal_params.queried_inds
        lcbal_p_train = config.lcbal_params.p_all[inds]
        lcbal_y_train = config.lcbal_params.y_all[inds]
        lcbal_X_train = config.lcbal_params.X_all[inds]
        p_adjusted = lcbal_p_train / np.max(lcbal_p_train)  # TODO use a value other than the max? consider the approach used by Alejandro Correa Bahnsen http://albahnsen.github.io/CostSensitiveClassification/Sampling.html
        # apply rejection sampling
        # for notes and citations, see markdown cell in BatchResultAnalysis notebook
        sample_rand = np.random.random_sample(len(lcbal_y_train))
        to_include = sample_rand <= p_adjusted
        lcbal_X_train = lcbal_X_train[to_include]
        lcbal_y_train = lcbal_y_train[to_include]
        n_pos = np.sum(lcbal_y_train)
        if config.lcbal_verbose:
            logger.debug(f"LCB-AL Rejection Sampling: {np.sum(to_include)} / {len(p_adjusted)} included in training data (n_pos = {np.sum(lcbal_y_train)} / {len(lcbal_y_train)})")
        
        # train the classifier
        if n_pos >= 1 and n_pos < len(lcbal_y_train):
            clf, _ = alsim.classification.train_model(lcbal_X_train, lcbal_y_train, X_unlabeled_validation, config)
        elif config.lcbal_verbose:
            logger.debug("Skipping classifier training for data of a single class")
        
    # compute the true metrics on the held-out test set
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[:,1]
    true_metrics = alsim.metrics.compute_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    # we skim generating the estimated_metrics (and difference_metrics) until
    # we are at the 0-indexed batch number given by estimate_metrics_after_batch
    should_estimate_metrics = curr_batch >= config.estimate_metrics_after_batch
    # if an allow-list is provided, only estimate metrics 
    # during the configured list of batches
    if config.estimate_metrics_batch_allowlist:
        should_estimate_metrics = curr_batch in config.estimate_metrics_batch_allowlist
    
    if not should_estimate_metrics:
        # return only the true metrics
        model_metrics = {
            'n_train': len(y_train),
            'n_train_pos': np.sum(y_train),
            'true_metrics': true_metrics,
        }
        return model_metrics, new_classifier
    
    # now produce estimates for the metrics using the labeled data
    estimated_metrics = None
    if metric_strategy == 'random_only_cv':
        randomly_sampled_docs = np.array([doc['sample_type'] == 'random' 
                                          for doc in selected_training_docs])
        if np.sum(randomly_sampled_docs) <= 1:
            # 0 or 1 randomly sampled docs, so back off to regular cross-validation
            metric_strategy = 'cv'
        else:  # at least 1 randomly-sampled doc
            randomly_sampled_doc_inds = randomly_sampled_docs.nonzero()[0]
            X_full = X_train
            y_full = y_train
            all_labeled_inds = np.arange(len(y_full))
            y_random = y_full[randomly_sampled_docs]
            random_pos_count = np.sum(y_random)
            if random_pos_count == 0 or random_pos_count == len(y_full):
                # back off to CV when there are no examples of one of the classes
                # note this is quite likely in a "skew" environment
                metric_strategy = 'cv'
            else:  # at least 1 positive and 1 negative 
                y_random_pred = np.zeros(len(randomly_sampled_doc_inds))
                y_random_pred_proba = np.zeros(len(randomly_sampled_doc_inds), dtype=float)
                for i, test_ind in enumerate(randomly_sampled_doc_inds):
                    train_inds = all_labeled_inds != test_ind
                    X_train, X_test = X_full[train_inds], X_full[test_ind]
                    y_train, y_test = y_full[train_inds], y_full[test_ind]
                    #assert len(y_train) == len(y_full) - 1, f"{y_train.shape} {test_ind}"
                    loo_clf, unlabeled_valid_pct_pos = alsim.classification.train_model(X_train, y_train, X_unlabeled_validation, config)
                    unlabeled_valid_pct_pos_list.append(unlabeled_valid_pct_pos)
                    y_pred = loo_clf.predict(np.reshape(X_test, (1, -1)))
                    y_pred_proba = loo_clf.predict_proba(np.reshape(X_test, (1, -1)))[:,1]
                    y_random_pred[i] = y_pred
                    y_random_pred_proba[i] = y_pred_proba
                estimated_metrics = alsim.metrics.compute_metrics(y_random, y_random_pred, y_random_pred_proba)
    if metric_strategy == 'cv':
        # estimate metrics using cross validation over all of the data 
        n = len(y_train)
        
        X_full = X_train
        y_full = y_train
        y_full_pred = np.zeros_like(y_full)
        y_full_pred_proba = np.zeros_like(y_full, dtype=float)
        n_splits = min(n, 50)  # defines the maximum number of folds
        kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=False)
        for train_index, test_index in kf.split(X_full):
            X_train, X_test = X_full[train_index], X_full[test_index]
            y_train, y_test = y_full[train_index], y_full[test_index]
            pos_count = np.sum(y_train)
            if pos_count == 0:  
                # the fold contains no positive training examples
                # so predict negative class
                y_pred = np.zeros_like(y_test)
                y_pred_proba = np.zeros_like(y_test, dtype=float)
            elif pos_count == len(y_train):
                # the fold contains only positive training examples
                # so predict positive class
                y_pred = np.ones_like(y_test)
                y_pred_proba = np.ones_like(y_test, dtype=float)
            else:  # 1+ pos and neg training examples
                loo_clf, unlabeled_valid_pct_pos = alsim.classification.train_model(X_train, y_train, X_unlabeled_validation, config)
                y_pred = loo_clf.predict(X_test)
                y_pred_proba = loo_clf.predict_proba(X_test)[:,1]
                unlabeled_valid_pct_pos_list.append(unlabeled_valid_pct_pos)
            y_full_pred[test_index] = y_pred
            y_full_pred_proba[test_index] = y_pred_proba
        # we have now produced CV predictions for all the labeled data,
        # so we can compute the estimated model performance metrics
        estimated_metrics = alsim.metrics.compute_metrics(y_full, y_full_pred, y_full_pred_proba)
        
    if metric_strategy == 'bootstrap_merge':
        # this is a deprecated bootstrap implementation
        # it is a mistaken implementation of hanczar_small-sample_2010 that merges after computing scores, rather than maintaining samples and computing once
        B = 100  # number of bootstraps
        
        n = len(y_train)
        X_full = X_train
        y_full = y_train
        
        train_estimated_metrics_list = []
        test_estimated_metrics_list = []
        available_inds = np.array(range(n))
        available_inds_set = set(available_inds)
        while len(train_estimated_metrics_list) < B:
            # sample with replacement
            train_inds = np.random.choice(available_inds, size=n)
            X_train, y_train = X_full[train_inds], y_full[train_inds]
            test_inds = list(available_inds_set - set(train_inds))
            X_test, y_test = X_full[test_inds], y_full[test_inds]
            
            train_pos_count = np.sum(y_train)
            test_pos_count = np.sum(y_test)
            if train_pos_count == 0 or train_pos_count == len(y_train) or test_pos_count == 0 or test_pos_count == len(y_test):
                # no or all positives in either the train or test fold
                # therefore, skip this bootstrap sample
                continue
            bootstrap_clf, unlabeled_valid_pct_pos = train_model(X_train, y_train, X_unlabeled_validation, config)
            unlabeled_valid_pct_pos_list.append(unlabeled_valid_pct_pos)
            y_train_pred = bootstrap_clf.predict(X_train)
            y_train_pred_proba = bootstrap_clf.predict_proba(X_train)[:,1]
            y_test_pred = bootstrap_clf.predict(X_test)
            y_test_pred_proba = bootstrap_clf.predict_proba(X_test)[:,1]
            train_metrics = compute_metrics(y_train, y_train_pred, y_train_pred_proba)
            test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)
            train_estimated_metrics_list.append(train_metrics)
            test_estimated_metrics_list.append(test_metrics)
        # now combine the estimated metrics into various bootstraps
        def combine_bootstrap_metrics(metrics_list):
            assert len(metrics_list) > 0
            keys = metrics_list[0].keys()
            sums = defaultdict(float)
            for metrics in metrics_list:
                for key in keys:
                    sums[key] += metrics[key]
            mean_metrics = {key: sums[key] / len(metrics_list) for key in keys}
            return mean_metrics
        train_metrics = combine_bootstrap_metrics(train_estimated_metrics_list)
        test_metrics = combine_bootstrap_metrics(test_estimated_metrics_list)
        # 0.632 estimator
        # magic numbers are approximations for 1/e and 1 - 1/e
        b632_metrics = {}
        for key in test_metrics.keys():
            b632_metrics[key] = 0.632 * test_metrics[key] + 0.368 * train_metrics[key]
        # save references for later
        estimated_metrics = b632_metrics
        bootstrap_loo_estimated_metrics = test_metrics
    if metric_strategy == 'bootstrap':
        B = 100  # number of bootstraps
        
        n = len(y_train)
        X_full = X_train
        y_full = y_train
        
        n_bootstraps_computed = 0
        y_train_list = []
        y_train_pred_list = []
        y_train_pred_proba_list = []
        y_test_list = []
        y_test_pred_list = []
        y_test_pred_proba_list = []
        
        available_inds = np.array(range(n))
        available_inds_set = set(available_inds)
        while n_bootstraps_computed < B:
            # sample with replacement
            train_inds = np.random.choice(available_inds, size=n)
            X_train, y_train = X_full[train_inds], y_full[train_inds]
            test_inds = list(available_inds_set - set(train_inds))
            X_test, y_test = X_full[test_inds], y_full[test_inds]
            
            train_pos_count = np.sum(y_train)
            test_pos_count = np.sum(y_test)
            if train_pos_count == 0 or train_pos_count == len(y_train) or test_pos_count == 0 or test_pos_count == len(y_test):
                # no or all positives in either the train or test fold
                # therefore, skip this bootstrap sample
                continue
            else:
                n_bootstraps_computed += 1
            bootstrap_clf, unlabeled_valid_pct_pos = train_model(X_train, y_train, X_unlabeled_validation, config)
            unlabeled_valid_pct_pos_list.append(unlabeled_valid_pct_pos)
            y_train_pred = bootstrap_clf.predict(X_train)
            y_train_pred_proba = bootstrap_clf.predict_proba(X_train)[:,1]
            y_test_pred = bootstrap_clf.predict(X_test)
            y_test_pred_proba = bootstrap_clf.predict_proba(X_test)[:,1]
            
            y_train_list.append(y_train)
            y_train_pred_list.append(y_train_pred)
            y_train_pred_proba_list.append(y_train_pred_proba)
            y_test_list.append(y_test)
            y_test_pred_list.append(y_test_pred)
            y_test_pred_proba_list.append(y_test_pred_proba)
            
        y_train = np.concatenate(y_train_list)
        y_train_pred = np.concatenate(y_train_pred_list)
        y_train_pred_proba = np.concatenate(y_train_pred_proba_list)
        y_test = np.concatenate(y_test_list)
        y_test_pred = np.concatenate(y_test_pred_list)
        y_test_pred_proba = np.concatenate(y_test_pred_proba_list)
        # train_metrics is the resubstitution estimator
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_pred_proba)
        # test_metrics is the out-of-bag/LOO estimator
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)
        # b632_metrics is the 0.632 estimator
        # magic numbers are approximations for 1/e and 1 - 1/e
        b632_metrics = {}
        for key in test_metrics.keys():
            b632_metrics[key] = 0.632 * test_metrics[key] + 0.368 * train_metrics[key]
        # save references for later
        estimated_metrics = b632_metrics
        bootstrap_loo_estimated_metrics = test_metrics
        
    # compute predicted proportion positive based on the overall model and any CV folds that contributed to it
    if len(unlabeled_valid_pct_pos_list) > 0:
        mean_validation_pred_pos_pct = np.mean(unlabeled_valid_pct_pos_list)
        ci = alsim.utils.compute_ci(len(unlabeled_valid_pct_pos_list), np.std(unlabeled_valid_pct_pos_list))
        estimated_metrics['validation_pred_pos_pct_mean'] = mean_validation_pred_pos_pct
        estimated_metrics['validation_pred_pos_pct_upper'] = min(mean_validation_pred_pos_pct + ci, 0.99999)
        estimated_metrics['validation_pred_pos_pct_lower'] = max(mean_validation_pred_pos_pct - ci, 0.00001)
    if estimated_metrics is None:
        raise ValueError(f"Estimation strategy '{metric_strategy}' not implemented.")
    
    # compute divergence between estimated and true metrics
    difference_metrics = {}
    for key in estimated_metrics.keys():
        if key in true_metrics:
            difference_metrics[key] = estimated_metrics[key] - true_metrics[key]
    
    # package up the various types of metrics (estimates vs hold-out test, and their differences)
    model_metrics = {
        'n_train': len(y_train),
        'n_train_pos': np.sum(y_train),
        'true_metrics': true_metrics,
        'estimated_metrics': estimated_metrics,
        'difference_metrics': difference_metrics,
    }
    if metric_strategy == 'bootstrap':
        # bootstrapping is super expensive, so rather than waste time doing multiple experiments we return multiple versions of the difference metrics
        # currently, that means that difference_metrics contains the 632 estimator, while bootstrap_loo_difference_metrics contains the 'leave-one-out' (out-of-bag) estimator
        model_metrics['bootstrap_loo_estimated_metrics'] = bootstrap_loo_estimated_metrics
        bootstrap_loo_difference_metrics = {}
        for key in bootstrap_loo_estimated_metrics.keys():
            if key in true_metrics:
                bootstrap_loo_difference_metrics[key] = bootstrap_loo_estimated_metrics[key] - true_metrics[key]
        model_metrics['bootstrap_loo_difference_metrics'] = bootstrap_loo_difference_metrics
    
    return model_metrics, new_classifier
    
    
def train_and_evaluate_model_config(config, training_docs, testing_docs):
    run_metrics_list = []
    overall_start_time = datetime.now()
    model_dict = {
        'config': config,
        'overall_start_time': str(overall_start_time),
        'run_metrics_list': run_metrics_list,
    }    
    for run_ind in tqdm(range(config.n_runs), disable=not config.should_tqdm_runs):
    
        all_training_docs = training_docs[:]
        random.shuffle(all_training_docs)

        # hold-out some of the training docs as validation
        pct_validation = 0.2
        pct_training = 1 - pct_validation
        n_training = int(len(all_training_docs) * pct_training)
        training_pool_docs = all_training_docs[:n_training]
        validation_pool_docs = all_training_docs[n_training:]
        assert len(training_pool_docs) > len(validation_pool_docs)
        assert len(training_pool_docs) >= 1000, "Something weird is happening with the availability of training data; check inputs."
        
        # clean out any temporary keys stored in the docs
        # hacky
        for doc in training_pool_docs:
            if 'is_kmeans_sample_eligible' in doc:
                del doc['is_kmeans_sample_eligible']
        
        # these lists track progress over the run
        batch_metrics_list = []
        run_metrics_list.append(batch_metrics_list)
        
        selected_training_docs = []
        existing_classifier = None

        n_batches = config.n_batches 
        for batch in range(n_batches):
            batch_start = datetime.now()

            # get a new sample of training data
            prev_batch_metrics = batch_metrics_list[-1] if len(batch_metrics_list) > 0 else {}
            selected_inds = alsim.sample.get_batch_sample(config, config.batch_size, batch, config.n_batches, selected_training_docs, training_pool_docs, prev_batch_metrics, existing_classifier)
            
            # update the selected_training_docs with newly selected docs
            original_labeled_size = len(selected_training_docs)
            for selected_ind_tup in sorted(selected_inds, reverse=True, key=lambda tup: tup[0]):
                selected_ind, sample_type = selected_ind_tup
                doc = training_pool_docs[selected_ind]
                doc['sample_type'] = sample_type
                del training_pool_docs[selected_ind]
                selected_training_docs.append(doc)
            new_labeled_size = len(selected_training_docs)
            assert original_labeled_size + len(selected_inds) == new_labeled_size

            # train model and get evaluation
            batch_metrics, new_classifier = train_and_evaluate_model(
                batch,
                config, 
                selected_training_docs, 
                validation_pool_docs, 
                testing_docs
            )
            if new_classifier is not None:
                existing_classifier = new_classifier
                
            batch_metrics['total_time_seconds'] = (datetime.now() - batch_start).total_seconds()
            batch_metrics_list.append(batch_metrics)
            
        # do between-run cleanup here
        if config.sample_strategy == 'lcbal':
            del config.lcbal_params
    
    model_dict['combined_metrics_list'] = alsim.metrics.combine_run_metrics(run_metrics_list)
    
    model_dict['overall_elapsed_time_seconds'] = (datetime.now() - overall_start_time).total_seconds()
    
    return model_dict
