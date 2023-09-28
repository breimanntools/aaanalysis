"""
This is a script for the naming clusters method of AAclust.
"""
import time
import pandas as pd
import aaanalysis.utils as ut

# I Helper function
def _get_cluster_names(list_names=None, name_medoid=None, name_unclassified="Unclassified"):
    """
    Get list of cluster names sorted based on following criteria (descending order):
        a) Frequency of term (most frequent term is preferred)
        b) Term is the name or a sub-name of the given medoid
        c) Length of term (shorter terms are preferred)
    If cluster consists of only one term, the name will be 'unclassified ('category name')'
    """
    def remove_2nd_info(name_):
        """Remove information given behind comma"""
        if "," in name_:
            name_ = name_.split(",")[0]
            if "(" in name_:
                name_ += ")"  # Close parenthesis if interpreted by deletion
        return name_
    # Filter categories (Remove unclassified scales and secondary infos)
    list_names = [remove_2nd_info(x) for x in list_names if ut.STR_UNCLASSIFIED not in x]
    # Create list of shorter names not containing information given in parentheses
    list_short_names = [x.split(" (")[0] for x in list_names if " (" in x]
    if len(list_names) > 1:
        list_names.extend(list_short_names)
        # Obtain information to check criteria for sorting scale names
        df_counts = pd.Series(list_names).value_counts().to_frame().reset_index()   # Compute frequencies of names
        df_counts.columns = ["name", "count"]
        df_counts["medoid"] = [True if x in name_medoid else False for x in df_counts["name"]]  # Name in medoid
        df_counts["length"] = [len(x) for x in df_counts["name"]]      # Length of name
        # Sort names based on given criteria
        df_counts = df_counts.sort_values(by=["count", "medoid", "length"], ascending=[False, False, True])
        names_cluster = df_counts["name"].tolist()
    else:
        names_cluster = [name_unclassified]
    return names_cluster

# II Main function
def name_clusters(names=None, labels=None, dict_medoids=None):
    """"""
    # Get cluster labels sorted in descending order of frequency
    labels_sorted = pd.Series(labels).value_counts().index
    # Assign names to cluster
    dict_cluster_names = {}
    for clust in labels_sorted:
        name_medoid = names[dict_medoids[clust]]
        list_names = [names[i] for i in range(0, len(names)) if labels[i] == clust]
        names_cluster = _get_cluster_names(list_names=list_names, name_medoid=name_medoid,
                                           name_unclassified=ut.STR_UNCLASSIFIED)
        assigned = False
        for name in names_cluster:
            if name not in dict_cluster_names.values() or name == ut.STR_UNCLASSIFIED:
                dict_cluster_names[clust] = name
                assigned = True
                break
        if not assigned:
            dict_cluster_names[clust] = ut.STR_UNCLASSIFIED
    cluster_names = [dict_cluster_names[label] for label in labels]
    return cluster_names