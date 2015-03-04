from datetime import datetime
import random

class Point:
  '''
  utility class for storing a location and the cluster to which it belongs
  '''

  def __init__(self, position):
    ''' creates a point at the specified position, with cluster index set to 0 '''
    self._position = position
    self._index = 0

  def __str__(self):
    return str(self._position)

  def __repr__(self):
    return str(self)

  def reset(self):
    ''' moves the point to the origin '''
    self._position = [0 for i in self._position]

  def add(self, point):
    ''' adds another point's position to the position of this point (a translation) '''
    self._position = [sum(x) for x in zip(self._position, point._position)]

  def scale(self, scale):
    ''' scales the position of this point by a constant '''
    self._position = [x * scale for x in self._position]

  def clone(self):
    ''' return a new point at the same position as this point (without assigning the new point to a cluster) '''
    return Point([x for x in self._position])

class KMeans:
  '''
  k-means clustering!
  https://en.wikipedia.org/wiki/K-means_clustering
  '''

  @staticmethod
  def get_distance2(p1, p2):
    ''' returns squared Euclidean distance between two coordinates '''
    return sum([(a - b) ** 2 for (a, b) in zip(p1, p2)])

  def __init__(self, data, means=None, k=None, dist=None):
    '''
    initializes the classifier
    data: a list of coordinates
    means: optional list of means, defaults to the first k points in the data list
    k: optional number of means, defaults to the length of the list list means
    dist: optional distance function to use, defaults to squared Euclidean distance
    exactly one of means and k must be provided
    '''
    self._data = [Point(d) for d in data]
    self._dist = dist if dist is not None else KMeans.get_distance2
    if means is not None:
      self._means = [Point(m) for m in means]
      self._k = len(means)
    elif k is not None:
      self._k = k
      self._means = [Point(d) for d in data[:k]]
    else:
      raise Exception('exactly one of means or k must be provided')

  def solve(self, limit_iterations=None, limit_time=None):
    '''
    runs until convergence, the optional iteration limit is reached, or the optional time limit is reached
    '''
    updated = True
    iteration = 0
    if limit_time is not None:
      start_time = datetime.now()
      timer = 0
    while updated and (limit_iterations is None or iteration < limit_iterations) and (limit_time is None or timer < limit_time):
      updated = self.iterate()
      iteration += 1
      if limit_time is not None:
        timer = (datetime.now() - start_time).total_seconds()

  def iterate(self):
    ''' runs a single iteration '''
    # remember if there was an update
    updated = False
    # first, assign each data point to the nearest cluster
    for datum in self._data:
      cluster = self.classify(datum._position)
      if datum._index != cluster:
        updated = True
        datum._index = cluster
    # if there cluster assignments have changed, then the means need to be recalculated
    if updated:
      # recalculate means with the new cluster assignments
      for i in range(len(self._means)):
        num = 0
        for datum in self._data:
          if datum._index == i:
            if num == 0:
              self._means[i].reset()
            self._means[i].add(datum)
            num += 1
        if num > 0:
          self._means[i].scale(1 / num)
    # return whether or not the clusters were updated
    return updated

  def classify(self, point):
    ''' returns the index of the cluster to which the given point belongs '''
    cluster = 0
    min = 0
    for i in range(len(self._means)):
      distance = self._dist(self._means[i]._position, point)
      if i == 0 or distance < min:
        cluster = i
        min = distance
    return cluster

  def get_energy(self):
    ''' returns the cumulative distance of all points to means '''
    energy = 0
    for datum in self._data:
      energy += self._dist(datum._position, self._means[datum._index]._position)
    return energy

  def get_means(self):
    ''' returns a list coordinates of each of the means '''
    return [m._position for m in self._means]

if __name__ == '__main__':
  # example usage
  n = 10
  m = n // 2
  offset = 1.25
  data = []

  # class A
  for i in range(m):
    data.append([random.gauss(-offset, 1)])

  # class B
  for i in range(n - m):
    data.append([random.gauss(+offset, 1)])

  # cluster the points
  kmeans = KMeans(data, means=[[-offset], [+offset]])
  kmeans.solve()

  # show the results
  print('Cluster Means:')
  for mean in kmeans.get_means():
    print(' %+.3f'%(mean[0]))
  print('Classification:')
  correct = 0
  for (i, d) in enumerate(data):
    cluster = kmeans.classify(d)
    target = 0 if i < m else 1
    if cluster == target:
      correct += 1
    print(' [%+.3f] -> %d'%(d[0], cluster))
  print('Accuracy: %d/%d (%d%%)'%(correct, n, correct / n * 100))
