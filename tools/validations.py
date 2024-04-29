from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def ari_score(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)


def nmi_score(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)