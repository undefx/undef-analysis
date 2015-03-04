import random
import numpy as np

class PCA:
  '''
  Principal component analysis!
  https://en.wikipedia.org/wiki/Principal_component_analysis
  http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
  '''

  def __init__(self, data, dimensions):
    '''
    finds the principal components and projects the data onto a subset of these components
    data: a list of high dimensional points *assumed to be in Euclidean space*
    dimensions: the dimensionality of the output
    '''
    if dimensions > len(data[0]):
      raise Exception('output dimensions must be less than or equal to the number of input dimensions [%d > %d]'%(dimensions, len(data[0])))
    data = np.array(data).transpose()
    vals, vecs = np.linalg.eig(np.cov(data))
    pairs = [(abs(val), vec) for (val, vec) in zip(vals, vecs.transpose())]
    pairs.sort()
    pairs.reverse()
    total = sum([p[0] for p in pairs])
    self._components = [(p[0] / total, p[1].tolist()) for p in pairs]
    self._w = np.array([p[1] for p in pairs[:dimensions]])
    self._mean = np.array([np.mean(d) for d in data])
    self._result = [self.project(d) for d in data.transpose()]

  def project(self, vector):
    ''' projects a point onto a subset of the principal components '''
    return self._w.dot(vector - self._mean).tolist()

  def get_result(self):
    ''' returns the data projected onto a subset of the principal components '''
    return self._result

  def get_components(self):
    ''' returns the sorted list of all principal components '''
    return self._components

if __name__ == '__main__':
  # example usage
  input = [[i + random.gauss(0, 1), i + random.gauss(0, 1)] for i in range(5)]
  pca = PCA(input, 1)
  output = pca.get_result()
  component = pca.get_components()[0]
  print('The principle component is (%+.3f, %+.3f) and captures %d%% of the variance.'%(component[1][0], component[1][1], component[0] * 100))
  for (i, o) in zip(input, output):
    print('(%+.3f, %+.3f) -> (%+.3f)'%(i[0], i[1], o[0]))
