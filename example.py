import random
from mds import MDS
from pca import PCA
from kmeans import KMeans
import matplotlib.pyplot as plt

# test data
n = 10
m = n // 2
offset = 1
input = []
for i in range(m):
  input.append([i + random.gauss(0, 1), random.gauss(0, 1)])
for i in range(n - m):
  input.append([i + m + offset + random.gauss(0, 1), random.gauss(0, 1)])
means = [sum([i[x] for i in input]) / n for x in range(2)]
input = [[i[x] - means[x] for x in range(2)] for i in input]

# kmeans clustering
print('=== k-means ===')
kmeans = KMeans(input, k=2)
kmeans.solve()
print('Cluster Means:')
means = kmeans.get_means()
for mean in means:
  print(' (%+.3f, %+.3f)'%(mean[0], mean[1]))
print('Classification:')
correct = 0
cluster_colors = []
for (i, d) in enumerate(input):
  cluster = kmeans.classify(d)
  target = 0 if i < m else 1
  if cluster == target:
    correct += 1
  cluster_colors.append(cluster)
  print(' (%+.3f, %+.3f) -> %d'%(d[0], d[1], cluster))
print('Accuracy: %d/%d (%d%%)'%(correct, n, correct / n * 100))
print(cluster_colors)

# pca solution
print('=== PCA ===')
pca = PCA(input, 1)
pca_output = pca.get_result()
for (i, o) in zip(input, pca_output):
  print('(%+.3f, %+.3f) -> (%+.3f)'%(i[0], i[1], o[0]))

# mds solution
print('=== MDS ===')
mds = MDS(MDS.get_distances(input), 1)
mds_output = mds.solve()
for (i, o) in zip(input, mds_output):
  print('(%+.3f, %+.3f) -> (%+.3f)'%(i[0], i[1], o[0]))

# mds, given pca
print('=== PCA -> MDS ===')
mds = MDS(MDS.get_distances(input), 1)
pca_mds_output = mds.solve(guess=pca_output)
for (i, o) in zip(input, pca_mds_output):
  print('(%+.3f, %+.3f) -> (%+.3f)'%(i[0], i[1], o[0]))

# plot the results
rainbow = [i for i in range(n)]
plt.gca().set_autoscale_on(False)
plt.axis([-8, 8, -5, 5])
plt.scatter([p[0] for p in means], [p[1] for p in means], c=['#8080ff', '#ff8080'], s=500)
plt.scatter([p[0] for p in input], [p[1] for p in input], c=cluster_colors, s=150)
plt.scatter([p[0] for p in input], [p[1] for p in input], c=rainbow)
plt.scatter([p[0] for p in pca_output], [4.0 for i in range(n)], c=rainbow)
plt.scatter([p[0] for p in mds_output], [4.2 for i in range(n)], c=rainbow)
plt.scatter([p[0] for p in pca_mds_output], [4.4 for i in range(n)], c=rainbow)
plt.show()
