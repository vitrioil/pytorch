#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch {
namespace lazy {

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ltc_device_data,
          data->shape(),
          /*num_outputs=*/1,
          /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, ltc_device_data);
}

NodePtr DeviceData::Create(std::shared_ptr<BackendData> data) {
  NodePtr node = nullptr;
  if (FLAGS_torch_lazy_reuse_ir) {
    TORCH_LAZY_TIMED("DeviceData::ReuseNode");
    node = ReuseNode<DeviceData>(
            OpKind(ltc_device_data),
            data);
  }
  if (node) {
    // At this point, we know the to-be-reused node has the same shape
    // and is safe to reuse, however, we need to replace the old data_
    // with the new one. Ditching the old data_ is safe because tracing
    // is done iteration by iteration, and also by the time we are lauching
    // async device execution, DeviceData nodes are not needed anymore.
    // If we ever want to extend our tracing to be multi-threaded,
    // making nodes as thread_local should solve the problem.
    DeviceData* device_data = static_cast<DeviceData*>(node.get());
    device_data->SetData(data);
  } else {
    TORCH_LAZY_TIMED("DeviceData::ComputeShapeAndMakeNode");
    node = MakeNode<DeviceData>(data);
  }
  return node;
}

} // namespace lazy
} // namespace torch
