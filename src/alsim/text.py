import os
import re
import pickle
import numpy as np
import logging
import spacy
import scipy
import sklearn
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    CountVectorizer,
    TfidfVectorizer,
)

import alsim.paths

AMP_MAP = {  # used to clean up a few HTML entities still hanging around in the data
    "&quot;": '"',
    "&amp;": "&",
    "&eacute;": "e",
    "&eth;": "ð",
    "&egrave;": "e",
    "&apos;": "a",
    "&middot;": "·",
    "&laquo;": "<",
    "&raquo;": ">",
    "&aacute;": "a",
    "&iacute;": "i",
    "&lt;": "<",
    "&gt;": ">",
    "&ograve;": "o",
    "&Uuml;": "U",
}


def add_tokens(docs):
    """
    Mutates each dictionary in the given iterable to add 'tokens' and 'tokens_set' keys.

    Depends on the 'text' key being present in the document.
    """
    nlp = spacy.load("en_core_web_sm")
    spacy_tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

    texts = [d["text"] for d in docs]
    for text_dict, tokenized_text in zip(
        docs, spacy_tokenizer.pipe(texts, batch_size=len(docs))
    ):
        # we remove non-alpha characters and lowercase all tokens
        tokens = [re.sub("[^a-z]", "", token.text.lower()) for token in tokenized_text]
        tokens = [t for t in tokens if t != ""]
        text_dict["tokens"] = tokens
        text_dict["tokens_set"] = set(tokens)


def load_texts():
    fulltext_docs_filepath = os.path.join(
        alsim.paths.DERIVED_DATA_DIR, "amazon_reviews_us", "fulltext_docs.pkl"
    )
    with open(fulltext_docs_filepath, "rb") as infile:
        fulltext_docs = pickle.load(infile)
    logger = logging.getLogger("alsim.text.load_texts")
    logger.info(f"Loaded {len(fulltext_docs)} documents with texts.")
    return fulltext_docs


def join_text_to_docs(training_docs, testing_docs, fulltext_docs):
    # add text info to train and test docs
    text_map = {d["document_id"]: d["text"] for d in fulltext_docs}

    for docs_list in [training_docs, testing_docs]:
        for doc in docs_list:
            text = text_map[doc["document_id"]]
            text = text.replace("<br />", " ").replace("<BR>", " ")
            amps = re.findall("&[A-Za-z]*;", text)
            if len(amps) > 0:
                for amp in amps:
                    text = text.replace(amp, AMP_MAP[amp])
            doc["text"] = text


def add_text_data_to_docs(training_docs, testing_docs):
    fulltext_docs = load_texts()
    join_text_to_docs(training_docs, testing_docs, fulltext_docs)
    add_tokens(training_docs)
    add_tokens(testing_docs)


def precompute_text_vectors(config, training_docs, testing_docs):
    """
    Compute vector representations of all of the docs.
    """
    logger = logging.getLogger("alsim.text.precompute_text_vectors")
    if config.vector_source == "roberta_mean_pool":
        # assumed to be already present in docs
        return
    if config.vector_source == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=config.vector_ngram_range,
            max_features=config.vector_max_features,
            min_df=1,
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            token_pattern=None,
        )
    elif config.vector_source == "bow":
        vectorizer = CountVectorizer(
            ngram_range=config.vector_ngram_range,
            max_features=config.vector_max_features,
            min_df=1,
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            token_pattern=None,
        )
    elif config.vector_source == "hash":
        # feature hashing is annoying here, since for interface uniformity we create dense matrices
        raise ValueError("Not yet implemented.")
    else:
        raise ValueError(f"Unknown vector_source '{vector_source}'.")

    X_train = vectorizer.fit_transform([doc["tokens"] for doc in training_docs])
    logger.info(f"{type(vectorizer).__name__} n_features={len(vectorizer.vocabulary_)}")
    X_test = vectorizer.transform([doc["tokens"] for doc in testing_docs])

    scaler = None
    for X, docs in [(X_train, training_docs), (X_test, testing_docs)]:
        X = scipy.sparse.csr_matrix.tocsr(X)

        densify = True
        should_scale = False  # TODO this should probably be a config value
        if densify:
            X = X.todense()
            if should_scale:
                if scaler is None:
                    scaler = sklearn.preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)

        for i, doc in enumerate(docs):
            doc[config.vector_source] = X[i, :]

    should_scale = False  # TODO this should probably be a config value
    if should_scale:
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
