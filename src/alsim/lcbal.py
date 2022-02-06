
import sklearn
import torch
import numpy as np
import logging

import alsim.classification
import alsim.data


class TorchLcbAl(torch.nn.Module):
    def __init__(self, n_inputs):
        super(TorchLcbAl, self).__init__()
        self.fc = torch.nn.Linear(n_inputs, 1)
        torch.nn.init.uniform_(self.fc.weight, -0.01, 0.01) 
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        z = self.fc(x)
        return z

    
class LcbAlParams:
    def __init__(self, config, training_pool_docs):
        n = len(training_pool_docs)
        self.n = n
        self.p_min = 1 / (config.lcbal_p_scale * n)
        self.R_squared = 100  # TODO ideally, remove this
        
        # this is the full training pool at the first batch that LCB-AL is invoked
        self.X_all = alsim.classification.vectorize_docs(config.vector_source, training_pool_docs)
        self.y_all = alsim.data.get_labels(config.outcome, training_pool_docs)

        self.clf = sklearn.linear_model.LogisticRegression(solver='lbfgs', penalty='l2')
        self.clf.classes_ = np.array([0, 1])
        self.clf.intercept_ = 0.0
        self.clf.coef_ = np.zeros(self.X_all.shape[1]).astype(np.float32).reshape((1, -1))

        self.queried_inds = []
        self.unqueried_inds = list(np.arange(n))
        self.p_all = np.zeros(n).astype('float32')
        
        self.document_ids = [doc['document_id'] for doc in training_pool_docs]


def run_lcbal(config, curr_batch, n_to_sample, training_pool_docs):
    logger = logging.getLogger("alsim.lcbal.run_lcbal")
    t = curr_batch + 1  # curr_batch is 0-indexed, we need t to start at 1
    
    if 'lcbal_params' not in config.__dict__:
        # perform initial setup
        p = LcbAlParams(config, training_pool_docs)
        config.lcbal_params = p
        if config.lcbal_verbose:
            logger.debug(f"Initiating LCB-AL: {p.n=}; {t=}")
    else:
        p = config.lcbal_params
    
    C_t = np.sqrt(np.log(t)) / 10 / config.lcbal_C_scale
    if t == 1:
        lambda_t = 0
    else:
        #queried_prob_sum = 1 / np.sum(p_all[queried_inds])
        #lambda_t = (100 * n * t) / (queried_prob_sum)**(1/3)
        lambda_t = 0.001 * p.n * t
    if config.lcbal_verbose:
        logger.debug(f"lambda_t={lambda_t:.6f} C_t={C_t:.6f}")
        
    preds_t = p.clf.predict_proba(p.X_all)[:,1].reshape(-1)
    y_bar = (preds_t >= 0.5).astype(int)
    y_bar[p.queried_inds] = p.y_all[p.queried_inds]
    
    losses = np.log(1 + np.exp(-y_bar * preds_t))  # margin-based logistic loss
    p_t = p.p_min + (1 - p.n * p.p_min) * losses / np.sum(losses)
    assert np.isclose(np.sum(p_t), 1), f"sum of p_t = {np.sum(p_t):.6f}, expected to sum to 1"
    if config.lcbal_verbose:
        logger.debug(f"p: min={np.min(p_t):.5f}, mean={np.mean(p_t):.5f}, max={np.max(p_t):.5f}. {np.sum(y_bar)=}")
    
    p_unqueried = p_t[p.unqueried_inds]
    p_unqueried = p_unqueried / p_unqueried.sum()
    new_queried_inds = np.random.choice(p.unqueried_inds, size=n_to_sample, replace=False, p=p_unqueried)
    for new_ind in new_queried_inds:
        p.unqueried_inds.remove(new_ind)
        #assert new_ind not in p.queried_inds
    p.queried_inds.extend(new_queried_inds)
    
    # identify selected inds to return
    # we do this by mapping from the original training pool to the current training pool, using the document_id
    doc_id_to_ind_map = {doc['document_id']: ind for ind, doc in enumerate(training_pool_docs)}
    for doc_id in doc_id_to_ind_map.keys():
        assert doc_id in p.document_ids, "New doc_id showed up; panic!"
    selected_inds = []
    for new_ind in new_queried_inds:
        doc_id = p.document_ids[new_ind]
        selected_ind = doc_id_to_ind_map[doc_id]
        selected_inds.append(int(selected_ind))

    p_queried = p_t[new_queried_inds]
    p.p_all[new_queried_inds] = p_queried
    
    X_train = torch.from_numpy(p.X_all[p.queried_inds].astype('float32'))
    y_train = torch.from_numpy(p.y_all[p.queried_inds].astype('float32')).reshape((-1, 1))
    p_train = torch.from_numpy(p.p_all[p.queried_inds])
    
    lcb_al = TorchLcbAl(X_train.shape[1])
    lcb_al.train()
    opt = torch.optim.LBFGS(lcb_al.parameters(), max_iter=1000)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')  

    def compute_loss():
        opt.zero_grad()
        output = lcb_al(X_train)
        queried_losses = bce_loss(output, y_train)
        
        # this is the trained version
        #queried_losses = np.log(1 + np.exp(-y_train * output))
        L_hat = (1 / (p.n * t)) * torch.sum(queried_losses * (1 / p_train))
        
        prob_weighted_losses = torch.sum(torch.square(queried_losses) / torch.square(p_train))
        total_queried_losses = torch.sum(queried_losses)
        V_t = prob_weighted_losses - torch.square(total_queried_losses)
        LCB_t = L_hat - C_t * torch.sqrt(V_t)
        # per abernathy, ||h||^2 is just the inner product <h, h>
        self_concordant_barrier = -torch.log(p.R_squared - torch.inner(lcb_al.fc.weight, lcb_al.fc.weight))
        #self_concordant_barrier = torch.linalg.vector_norm(lcb_al.fc.weight)
        loss = LCB_t + lambda_t * self_concordant_barrier # + torch.linalg.vector_norm(lcb_al.fc.weight)
        
        loss.backward()
        return loss

    opt.step(compute_loss)  # get loss, use to update wts

    output = lcb_al(X_train)  # monitor loss
    loss_val = compute_loss() 
    loss = loss_val.item()

    # update the classifier
    if not np.any(np.isnan(torch.detach(lcb_al.fc.bias).numpy())):
        p.clf.intercept_ = torch.detach(lcb_al.fc.bias).numpy()
        p.clf.coef_ = torch.detach(lcb_al.fc.weight).numpy()
    
    # log progress
    weights = list(np.concatenate([torch.detach(lcb_al.fc.bias).numpy(), torch.detach(lcb_al.fc.weight).numpy().reshape(-1)]))
    #p.weight_history.append((t, list(queried_inds), torch.detach(lcb_al.fc.bias).numpy(), torch.detach(lcb_al.fc.weight).numpy()))
    n_iter = next(iter(opt.state.values()))['n_iter']
    if config.lcbal_verbose:
        logger.debug(f"t = {t:>4} n_queried = {len(p.queried_inds)}  loss = {loss:.4f}  weights={weights[:3]} n_iter={n_iter}")
    
    return selected_inds