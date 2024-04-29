from sklearn.cluster import MiniBatchKMeans, KMeans
import tools.timers as tm
from tqdm.auto import tqdm

def kmeans_init(n_clusters, batch_size = 0, random_state = 42, n_init = 5):

    if batch_size == 0:
        return KMeans(n_clusters=n_clusters, random_state=random_state)


    kmeans_minibactch = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    return kmeans

@tm.timer_decorator
def kmeans_teach(train_data, kmeans):
    kmeans.fit(train_data)
    return kmeans

@tm.timer_decorator
def kmeans_batching(trainloader, device, kmeans_minibactch, model):
  for batch, (X, y) in tqdm(enumerate(trainloader)):
    model_data = model(X.to(device))
    kmeans_minibactch = kmeans_minibactch.partial_fit(model_data[0:32,:].cpu().detach().numpy())
