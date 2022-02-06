import os
import json
import pickle
import numpy as np

import alsim.paths


def load_search_data():
    """
    Load the external search_data dictionary from a JSON file at a particular place relative to the working directory.

    The search_data dict has a deep nested structure:
    outcome:
      broad_results:
        ...
      narrow_results:
        ...
      human_results:
        search_query:
          bkm:
            ...
          bm25:
            ...
          rm3:
            document_id_list
            document_id_score_dict: (only in bm25 or rm3)
              document_id: score
    """
    search_data_filepath = os.path.join(
        alsim.paths.DERIVED_DATA_DIR, "query", "search_data_all.json"
    )
    with open(search_data_filepath, "r") as infile:
        search_data = json.load(infile)
    return search_data


def load_training_and_testing_docs():
    # load in the document data
    document_dicts_filepath = os.path.join(
        alsim.paths.DERIVED_DATA_DIR, "amazon_reviews_us", "eval_docs.pkl"
    )
    with open(document_dicts_filepath, "rb") as infile:
        document_dicts = pickle.load(infile)

    training_docs = document_dicts[:10000]
    testing_docs = document_dicts[10000:]

    return training_docs, testing_docs


def get_labels(outcome, text_docs):
    y = np.array([td["outcomes"][outcome] for td in text_docs], dtype=int)
    return y
