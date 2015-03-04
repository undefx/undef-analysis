from datetime import datetime

class Point:
  '''
  utility class for storing a location and the value of the objective function at that location
  location is an (n+1)-dimensional point that is part of a simplex in n-dimensional space
  '''

  # stores the total number of times the objective function has been evaluated
  _numEvaluations = 0

  def __init__(self, location):
    ''' creates a new point at the specified coordinates '''
    self._location = location
    self._value = None

  def __str__(self):
    return "Point" + str(self._location) + "=" + str(self._value)

  def __repr__(self):
    return str(self)

  def _set_value(self, objective):
    ''' assigns a value to this point by calling the objective function '''
    self._value = objective(self._location)
    Point._numEvaluations = Point._numEvaluations + 1

  def get_center(points):
    ''' finds the center of mass of a set of points '''
    center = [0 for i in range(len(points))]
    for point in points:
      for i in range(len(point._location)):
        center[i] += point._location[i]
    for i in range(len(points[0]._location)):
      center[i] /= len(points)
    return center

  def new_point(point, center, scale, objective):
    ''' creates a new point by moving one point in relation to another point '''
    location = [(center[i] + (scale * (center[i] - point._location[i]))) for i in range(len(point._location))]
    point = Point(tuple(location))
    point._set_value(objective)
    return point

  def get_num_evaluations():
    ''' returns the total number of times the objective function has been called '''
    return Point._numEvaluations

class NelderMead:
  '''
  Derivative-free optimization! (via the Nelder-Mead algorithm)
  https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
  '''

  def __init__(self, objective, limit_iterations = None, limit_value = None, limit_time = None, alpha = 1.0, gamma = 2.0, rho = -0.5, sigma = 0.5):
    if limit_iterations is None and limit_value is None and limit_time is None:
      raise Exception('at least one of (limit_iterations, limit_value, limit_time) must be given')
    # the objective function - takes a single tuple as an argument
    self._objective = objective
    # the maximum number of times to evaluate the objective function
    self._limit = limit_iterations
    # the target value - iteration will stop once this value has been found
    self._target = limit_value
    # the maximum time to spend solving
    self._timer = limit_time
    # the reflection coefficient (default 1)
    self._alpha = alpha
    # the expansion coefficient (default 2)
    self._gamma = gamma
    # the contraction coefficient (default -1/2)
    self._rho = rho
    # the reduction coefficient (default 1/2)
    self._sigma = sigma

  def get_simplex(self, numDimensions, centroid = (), radius = 1.0):
    ''' creates a simplex around some point using the specified radius '''
    if len(centroid) == 0:
      centroid = tuple([0 for i in range(numDimensions)])
    points = []
    for i in range(numDimensions + 1):
      if i < numDimensions:
        coords = [(centroid[j] + (radius if i == j else 0)) for j in range(numDimensions)]
      else:
        coords = [(-(numDimensions ** -.5) * radius) for j in range(numDimensions)]
      point = Point(tuple(coords))
      point._set_value(self._objective)
      points.append(point)
    return points;

  def run(self, simplex):
    ''' iterates until either the target is found, the evaluation limit is reached, or the time limit is exceeded '''
    start_time = datetime.now()
    current_timer = print_timer = 0
    Point._numEvaluations = 0
    while (self._limit is None or Point.get_num_evaluations() < self._limit) and (self._timer is None or current_timer < self._timer):
      list.sort(simplex, key = lambda point: point._value)
      best = simplex[0]
      worse = simplex[-2]
      worst = simplex[-1]
      center = Point.get_center(simplex[:-1])
      pointR = Point.new_point(worst, center, self._alpha, self._objective)
      if best._value <= pointR._value < worse._value:
        simplex[-1] = pointR
      elif pointR._value < best._value:
        pointE = Point.new_point(worst, center, self._gamma, self._objective)
        if pointE._value < pointR._value:
          simplex[-1] = pointE
        else:
          simplex[-1] = pointR
      else:
        pointC = Point.new_point(worst, center, self._rho, self._objective)
        if pointC._value < worst._value:
          simplex[-1] = pointC
        else:
          for i in range(len(simplex) - 1):
            i = i + 1
            simplex[i] = Point.new_point(simplex[i], best._location, self._sigma, self._objective)
      current_timer = (datetime.now() - start_time).total_seconds()
      li = self._limit if self._limit is not None else 0
      lv = self._target if self._target is not None else 0
      lt = self._timer if self._timer is not None else 0
      if current_timer >= print_timer + 1:
        print_timer = current_timer
        print('NM [%d -> %d] [%.3f -> %.3f] [%.1f -> %.1f]'%(Point.get_num_evaluations(), li, simplex[-1]._value, lv, current_timer, lt))
      if self._target is not None and simplex[-1]._value < self._target:
        break
    list.sort(simplex, key = lambda point: point._value)
    print('NM [%d -> %d] [%.3f -> %.3f] [%.1f -> %.1f]'%(Point.get_num_evaluations(), li, simplex[0]._value, lv, current_timer, lt))
    return simplex[0]

if __name__ == "__main__":
  # example usage
  def himmelblau(params):
    (x, y) = params
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
  solver = NelderMead(himmelblau, limit_iterations=100, limit_value=0.001, limit_time=1)
  simplex = solver.get_simplex(2, (0, 0), 0.1)
  best = solver.run(simplex)
  print('num evaluations:', Point.get_num_evaluations())
  print('best:', best)
