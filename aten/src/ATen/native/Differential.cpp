#include <ATen/Dispatch.h>
#include <ATen/native/Differential.h>
#include <ATen/native/TensorIterator.h>

namespace at {

namespace native {

DEFINE_DISPATCH(chebpoly_stub);

Tensor chebpoly(Tensor& self, uint32_t order) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return result;
}

Tensor& chebpoly_(Tensor& self, uint32_t order) {
  auto iter = TensorIterator::unary_op(self, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return self;
}

Tensor& chebpoly_out(Tensor& result, const Tensor& self, uint32_t order) {
  auto iter = TensorIterator::unary_op(result, self);
  chebpoly_stub(iter.device_type(), iter, order);
  return result;
}

} // namespace native
} // namespace at
