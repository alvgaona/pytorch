#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/Differential.h>


namespace at {
namespace native {
namespace {

template <typename scalar_t>
static void chebpoly_kernel(TensorIterator& iter, Scalar order) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "chebpoly_cpu", [&] {
    auto n = order.to<scalar_t>();
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return chebpoly_calc(a, n); }
    );
  });
}

} // namespace

REGISTER_DISPATCH(chebpoly_stub, &chebpoly_kernel);

} // namespace native
} // namespace at
