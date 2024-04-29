from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def ari_score(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)


def nmi_score(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)
    print("Report : ",
          classification_report(y_test, y_pred))
