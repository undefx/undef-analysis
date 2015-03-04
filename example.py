import random
from mds import MDS
from pca import PCA
import matplotlib.pyplot as plt

# test data
n = 10
input = [[i + random.gauss(0, 1), random.gauss(0, 1)] for i in range(n)]
means = [sum([i[x] for i in input]) / n for x in range(2)]
input = [[i[x] - means[x] for x in range(2)] for i in input]

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
plt.scatter([p[0] for p in input], [p[1] for p in input], c=rainbow)
plt.scatter([p[0] for p in pca_output], [4.0 for i in range(n)], c=rainbow)
plt.scatter([p[0] for p in mds_output], [4.2 for i in range(n)], c=rainbow)
plt.scatter([p[0] for p in pca_mds_output], [4.4 for i in range(n)], c=rainbow)
plt.show()
