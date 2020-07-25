#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>

namespace at {

struct TensorIterator;

namespace native {

static inline double chebpoly_calc(double x, uint32_t order) {
  double result;
  if (fabs(x) < 1) {
    result = std::cos(order * std::acos(x));
  } else if (x >= 1) {
    result = std::acosh(order * std::acosh(x));
  } else {
    result = std::pow(-1, order) * std::cosh(order * std::acosh(-x));
  }
  return result;
}

using chebpoly_fn = void (*)(TensorIterator&, uint32_t);

DECLARE_DISPATCH(chebpoly_fn, chebpoly_stub);

} // namespace native
} // namespace at
