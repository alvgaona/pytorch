#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Differential.h>


namespace at {
namespace native {
namespace {

static void chebpoly_kernel(TensorIterator& iter, const uint32_t order) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "chebpoly_cpu", [&] {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return chebpoly_calc(a, order); }
    );
  });
}

} // namespace

REGISTER_DISPATCH(chebpoly_stub, &chebpoly_kernel);

} // namespace native
} // namespace at
