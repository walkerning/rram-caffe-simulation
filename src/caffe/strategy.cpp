#include "caffe/strategy.hpp"

namespace caffe {
  template <typename Dtype>
  void ThresholdFailureStrategy<Dtype>::Apply() {
    const vector<Blob<Dtype>* >& net_weights = net_->learnable_params();
    for (int i = 0; i < net_weights.size(); i++) {
      Dtype* diff_data = net_weights[i]->mutable_cpu_diff();
      int count = net_weights[i]->count();
      for (int j = 0; j < count; j ++) {
 	if (diff_data[j] <= threshold_) {
	  diff_data[j] = 0;
	}
      }
    }
  }
  INSTANTIATE_CLASS(FailureStrategy);
  INSTANTIATE_CLASS(ThresholdFailureStrategy);
} // namespace caffe
