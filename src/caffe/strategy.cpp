#include "caffe/strategy.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {
  template <typename Dtype>
  void ThresholdFailureStrategy<Dtype>::Apply() {
    const vector<Blob<Dtype>* >& net_weights = net_->failure_learnable_params();
    for (int i = 0; i < net_weights.size(); i++) {
      // threshold for this param
      Dtype rate = net_->params_lr()[i] * dynamic_cast<SGDSolver<Dtype>* >(const_cast<Solver<Dtype>* >(solver_))->GetLearningRate();
      Dtype threshold = threshold_ * rate;
      Dtype* diff_data = net_weights[i]->mutable_cpu_diff();
      int count = net_weights[i]->count();
      for (int j = 0; j < count; j ++) {
	// LOG(INFO) << "learning_rate: " << dynamic_cast<SGDSolver<Dtype>* >(const_cast<Solver<Dtype>* >(solver_))->GetLearningRate()
	// 	  << " param_lr: " << net_->params_lr()[i]
	// 	  << " diff: " << diff_data[j] 
	// 	  << " diff/(learning_rate*param_lr):" << diff_data[j]/rate;
 	if (fabs(diff_data[j]) <= threshold) {
	  //LOG(INFO) << "diff " << diff_data[j] << " less than threshold " << threshold_ << ". set to 0";
	  diff_data[j] = 0;
	  //LOG(INFO) << "will set " << diff_data[j] << " to 0.";
	}
      }
    }
  }

  template <typename Dtype>
  void RemappingFailureStrategy<Dtype>::Apply() {
    const Dtype* iters_p = fail_iterations_[i]->cpu_data();
    const Dtype* value_p = fail_iterations_[i]->cpu_diff();
    //const Dtype* flag_p = 

    const vector<int> ids = net_->failure_learnable_layer_ids();
    for (int i = 0; i < ids.size(); i++) {
      if (strcmp(net_->layers_[ids[i]]->type(), "InnerProduct") == 0) {
	//value_p
      }
    }
  }

  INSTANTIATE_CLASS(FailureStrategy);
  INSTANTIATE_CLASS(ThresholdFailureStrategy);
  INSTANTIATE_CLASS(RemappingFailureStrategy);
} // namespace caffe
