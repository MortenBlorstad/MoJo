import torch


class Normalize:

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
    self._impl = impl
    self._decay = decay
    self._max = max
    self._stdeps = stdeps
    self._vareps = vareps
    self._mean = torch.tensor(0.0, dtype=torch.float64)
    self._sqrs = torch.tensor(0.0, dtype=torch.float64)
    self._step = torch.tensor(0, dtype=torch.int64)

  def __call__(self, values, update=True):
    update and self.update(values)
    return self.transform(values)

  def update(self, values):
    x = values.to(torch.float64)
    m = self._decay
    self._step += 1
    self._mean = m * self._mean + (1 - m) * x.mean()
    self._sqrs = m * self._sqrs + (1 - m) * (x ** 2).mean()

  def transform(self, values):
    correction = 1 - self._decay ** self._step
    mean = self._mean / correction
    var = (self._sqrs / correction) - mean ** 2
    if self._max > 0.0:
      scale = torch.rsqrt(
          max(var, 1 / self._max ** 2 + self._vareps) + self._stdeps)
    else:
      scale = torch.rsqrt(var + self._vareps) + self._stdeps
    if self._impl == 'off':
      pass
    elif self._impl == 'mean_std':
      values -= mean
      values *= scale
    elif self._impl == 'std':
      values *= scale
    else:
      raise NotImplementedError(self._impl)
    return values


x = torch.tensor([1,2,3,4,5], dtype=torch.float32)

norm = Normalize()

print(norm(x))
print(norm(x))
print(norm(x))
print(norm(x))

