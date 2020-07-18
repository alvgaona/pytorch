#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>

namespace at {

struct TensorIterator;

namespace native {

using chebpoly_fn = void (*)(TensorIterator&, Scalar, Scalar);

DECLARE_DISPATCH(chebpoly_fn, chebploy_stub);

static inline double chebpoly_calc(double x, int order) {
  double result;
  if (fabs(x) < 1) {
    result = cos(order * acos(x));
  } else if (x >= 1) {
    result = acosh(order * acosh(x));
  } else {
    result = pow(-1, order) * cosh(order * acosh(-x));
  }
  return result;
}

} // namespace native
} // namespace at
