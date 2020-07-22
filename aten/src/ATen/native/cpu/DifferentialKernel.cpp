#include <ATen/native/BinaryOps.h>

#include <cmath>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/native/Differential.h>

namespace at {
namespace native {

static void chebpoly_kernel(TensorIterator& iter, int order) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "chebpoly_cpu", [&] {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t {
          return chebpoly_calc(a, order);
        }
  });
}

REGISTER_DISPATCH(chebpoly_stub, &chebpoly_kernel);

} // namespace native
} // namespace at
