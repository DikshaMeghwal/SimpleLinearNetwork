# %%
import numpy as np

# %%
X = np.random.rand(100, 2)

X[35:65] = np.random.rand(30,2) * -2

X[65:] = np.random.rand(35,2) * 5

# %%
import matplotlib.pyplot as plt
K = 3

centroids = np.random.rand(3,2)*2
plt.scatter(X[:,0], X[:,1], s=5)
plt.scatter(centroids[:,0], centroids[:,1], c='green')

# %%
X = X[:, None, :]
centroids = centroids[None, :]
print(X.shape, centroids.shape)

# %%
num_epochs = 10

for idx in range(num_epochs):

    closest_centroid = np.argmin(((X - centroids)**2).sum(axis=-1),axis=1)

    for i in range(K):
        centroids[0,i] = X[closest_centroid==i].squeeze().mean(axis=0)
    plt.clf()
    plt.scatter(X[:,0,0], X[:,0,1], s=5)
    plt.scatter(centroids[0,:,0], centroids[0,:,1], c= 'green')
    plt.savefig(f"./K_means_{idx}.png")
    # print(centroids)
# %%

# %%
