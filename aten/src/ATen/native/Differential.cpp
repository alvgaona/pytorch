#include <ATen/native/Differential.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/core/DistributionsHelper.h>

namespace at { namespace native {

DEFINE_DISPATCH(chebpoly_stub);

Tensor chebpoly(const Tensor& self, order) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return result;
}

Tensor& chebpoly_(const Tensor& self, int order) {
  auto iter = TensorIterator::unary_op(self, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return self;
}

Tensor& chebpoly_out(Tensor& result, const Tensor& self, int order) {
  auto iter = TensorIterator::unary_op(result, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return result;
}
} // namespace native
} // namespace at