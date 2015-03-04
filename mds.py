import random
from neldermead import NelderMead

class MDS:
  '''
  Multidimensional scaling!
  https://en.wikipedia.org/wiki/Multidimensional_scaling
  '''

  @staticmethod
  def _distance(a, b):
    ''' returns the Euclidean distance between two vectors '''
    return sum([(x - y) ** 2 for (x, y) in zip(a, b)]) ** 0.5
    
  @staticmethod
  def get_distances(data, dist=None):
    ''' returns a square distance matrix, using an optional distance function (defaults to Euclidean distance) '''
    if dist is None:
      dist = MDS._distance
    return [[dist(a, b) for b in data] for a in data]

  def __init__(self, distances, ndim=2):
    '''
    distances: an N x N matrix of pairwise distances
    ndim: number of output dimensions, default 2
    '''
    self.distances = distances
    self.ndim = ndim
    self.npoints = len(distances)

  def solve(self, limit_iterations=1000, limit_time=5, guess=None):
    '''
    uses the Nelder-Mead algorithm to optimize an embedding in self.ndim dimensions
    limit: maximum number of iterations, default 1000
    guess: a list of self.npoints points as an initial guess, optional
    returns the list of best-fit points
    '''
    def objective(params):
      pts = [params[i * self.ndim : (i + 1) * self.ndim] for i in range(self.npoints)]
      sum = 0
      for i in range(self.npoints):
        for j in range(self.npoints):
          if i == j: continue
          mds_dist = MDS._distance(pts[i], pts[j])
          true_dist = self.distances[i][j]
          diff = abs(mds_dist - true_dist)
          sum += diff ** 2
      return sum
    if guess is None:
      initial = [random.gauss(0, 1) for i in range(self.npoints * self.ndim)]
    else:
      initial = []
      for i in range(self.npoints):
        for j in range(self.ndim):
          initial.append(guess[i][j])
    solver = NelderMead(objective, limit_iterations=limit_iterations, limit_time=limit_time)
    simplex = solver.get_simplex(len(initial), tuple(initial), 0.1)
    best = solver.run(simplex)
    final = best._location
    return [final[i * self.ndim : (i + 1) * self.ndim] for i in range(self.npoints)]

if __name__ == '__main__':
  # example usage
  input = [[i + random.gauss(0, 1), i + random.gauss(0, 1)] for i in range(5)]
  mds = MDS(MDS.get_distances(input), 1)
  output = mds.solve()
  for (i, o) in zip(input, output):
    print('(%+.3f, %+.3f) -> (%+.3f)'%(i[0], i[1], o[0]))
