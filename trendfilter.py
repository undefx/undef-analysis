# imports
import statistics
import cvxopt as co
co.solvers.options['show_progress'] = False

# discrete difference operator
def difference_operator(n, kp1, x=None):
  k = kp1 - 1
  if k < 0: raise Exception('k < 0')
  if kp1 >= n: raise Exception('kp1 >= n')
  if x is None:
    x = [i for i in range(n)]
  m = [[0 for col in range(n - k)] for row in range(n - k - 1)]
  if k == 0:
    for row in range(n - k - 1):
      m[row][row] = -1
      m[row][row + 1] = +1
    return m
  else:
    A = co.matrix(difference_operator(n - k, 1, x)).T
    B = co.matrix([[k / (x[i + k] - x[i]) if i == j else 0 for i in range(A.size[1])] for j in range(A.size[1])])
    C = co.matrix(difference_operator(n, k, x)).T
    M = A * B * C
    m = [[M.T[row * M.size[1] + col] for col in range(M.size[1])] for row in range(M.size[0])]
    return m

# trend filter implementation
def trend_filter(values, lambda_, order, positions=None):
  corr = co.matrix(values)
  n = len(values)
  m = n - order - 1
  D = co.matrix(difference_operator(n, order + 1, positions)).T
  P = D * D.T
  q = -D * corr
  G = co.spmatrix([], [], [], (2*m, m))
  G[:m, :m] = co.spmatrix(+1.0, range(m), range(m))
  G[m:, :m] = co.spmatrix(-1.0, range(m), range(m))
  h = co.matrix(float(lambda_), (2*m, 1))
  res = co.solvers.qp(P, q, G, h)
  points = corr - D.T * res['x']
  knots = [(i, x) for (i, x) in enumerate([y for y in D * points]) if abs(x) >= 1e-3]
  fit = [x for x in points]
  error = sum([(a - b) ** 2 for (a, b) in zip(values, fit)]) ** 0.5
  return fit, error, knots

# find trend filter lambda minimizing fitting error for some number of knots
def find_lambda(values, order, num_knots, lambda_max, threshold=1e-3):
  if num_knots == len(values) - order - 1:
    fit, error, knots = trend_filter(values, 0, order)
    return fit, error, knots, 0
  lambda_min = 0
  best = None
  count = 0
  while count < 100 and (best is None or lambda_max - lambda_min > threshold):
    lambda_ = (lambda_min + lambda_max) / 2
    fit, error, knots = trend_filter(values, lambda_, order)
    if len(knots) > num_knots:
      lambda_min = lambda_
    else:
      lambda_max = lambda_
      if len(knots) == num_knots:
        best = (fit, error, knots, lambda_max)
    count += 1
  return best

# trend filter path
def trend_filter_path(values, order):
  min_knots = 0
  max_knots = len(values) - order - 1
  # find a high enough value of lambda such that min_knots is reached
  lambda_max = 1
  fit, error, knots = trend_filter(values, lambda_max, order)
  while len(knots) > min_knots:
    lambda_max *= 2
    fit, error, knots = trend_filter(values, lambda_max, order)
  # fill the the path from min_knots to max_knots (decreasing lambda)
  path = []
  for num_knots in range(min_knots, max_knots + 1):
    result = find_lambda(values, order, num_knots, lambda_max, threshold=1e-3)
    lambda_max = result[-1]
    path.append(result)
  return path[::-1]

# cross-validated selection of trend filtering path
def cross_validated_trend_filter(values, order, k=5):
  path = trend_filter_path(values, order)
  error_curve = []
  for tf in path:
    fit, error, knots, lambda_ = tf
    num_knots = len(knots)
    errors = []
    for fold in range(k):
      val, pos = [], []
      for (i, v) in enumerate(values):
        if i == 0 or i == len(values) - 1 or (i - 1 + fold) % k != 0:
          val.append(v)
          pos.append(i)
      fit, error, knots = trend_filter(val, lambda_, order, positions=pos)
      test = []
      vals = []
      i, j = 0, 0
      while i < len(values):
        if pos[j] == i:
          j += 1
        else:
          vals.append(values[i])
          test.append((fit[j - 1] + fit[j]) / 2)
        i += 1
      error = sum([(a - b) ** 2 for (a, b) in zip(vals, test)]) ** 0.5
      errors.append(error)
    error_avg, error_se = statistics.mean(errors), statistics.pstdev(errors) / (k ** 0.5)
    error_curve.append({'tf': tf, 'num_knots': num_knots, 'lambda': lambda_, 'error': error_avg, 'se': error_se})
  error_min = None
  for e in error_curve:
    if error_min is None or e['error'] < error_min['error']:
      error_min = e
  error_1se = None
  for e in error_curve:
    if e['error'] < error_min['error'] + error_min['se']:
      error_1se = e
  return error_curve, error_min['tf'], error_1se['tf']
