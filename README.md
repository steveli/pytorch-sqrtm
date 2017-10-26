# Matrix square root for PyTorch

A PyTorch function to compute the square root of a matrix with gradient support.
The input matrix is assumed to be positive definite as matrix square root
is not differentiable for matrices with zero eigenvalues.


## Dependency

* [PyTorch](http://pytorch.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

## Example

```python
import torch
from torch.autograd import Variable
from sqrtm import sqrtm

k = torch.randn(20, 10)
# Create a (hopefully) positive definite matrix
pd_mat = Variable(k.t().matmul(k), requires_grad=True)
sqrt_mat = sqrtm(pd_mat)
sqrt_mat.sum().backward()
```
