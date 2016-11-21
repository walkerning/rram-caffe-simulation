#include "caffe/strategy.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {
  template <typename Dtype>
  void ThresholdFailureStrategy<Dtype>::Apply() {
    const vector<Blob<Dtype>* >& net_weights = net_->failure_learnable_params();
    int cleared = 0;
    int whole_num = 0;
    for (int i = 0; i < net_weights.size(); i++) {
      // threshold for this param
      Dtype rate = net_->params_lr()[i] * dynamic_cast<SGDSolver<Dtype>* >(const_cast<Solver<Dtype>* >(solver_))->GetLearningRate();
      Dtype threshold = threshold_ * rate;
      Dtype* diff_data = net_weights[i]->mutable_cpu_diff();
      int count = net_weights[i]->count();
      whole_num += count;
      for (int j = 0; j < count; j ++) {
	// LOG(INFO) << "learning_rate: " << dynamic_cast<SGDSolver<Dtype>* >(const_cast<Solver<Dtype>* >(solver_))->GetLearningRate()
	// 	  << " param_lr: " << net_->params_lr()[i]
	// 	  << " diff: " << diff_data[j] 
	// 	  << " diff/(learning_rate*param_lr):" << diff_data[j]/rate;
 	if (fabs(diff_data[j]) <= threshold) {
	  //LOG(INFO) << "diff " << diff_data[j] << " less than threshold " << threshold_ << ". set to 0";
	  cleared += 1;
	  diff_data[j] = 0;
	  //LOG(INFO) << "will set " << diff_data[j] << " to 0.";
	}
      }
    }
    // for get the ratio of threshold pruning
    // LOG(INFO) << "threshold strategy set " << float(cleared)/whole_num << " diff to zero";
  }

  template <typename Dtype>
  void RemappingFailureStrategy<Dtype>::GetFailFlagMat(Blob<Dtype>* failure_blob, shared_ptr<Blob<Dtype> > flag_mat_p) {
    const Dtype* iters_p = failure_blob->cpu_data();
    const Dtype* values_p = failure_blob->cpu_diff();
    Dtype* flag_p = flag_mat_p->mutable_cpu_data();
    for (int i = 0; i < failure_blob->count(); i ++) {
      if (iters_p[i] < 0 && values_p[i] == 0) {
	flag_p[i] = 1; // when this cell fail and stuck at 0, set the flag to 1
      }
    }
  }

  template <typename Dtype>
  void RemappingFailureStrategy<Dtype>::SortFCNeurons(vector<vector<int> >& orders) {
    // neurons in the first layer are not sorted //  according to output weight failures
    int size = net_->fc_params_ids_.size();
    vector<shared_ptr<Blob<Dtype> > > flag_mat_vec;
    for (int i = 0; i < size; i++) {
      shared_ptr<Blob<Dtype> > flag_mat_p(new Blob<Dtype>()); // consist of 0 and 1, 1 if fail, 0 if not
      //Blob<Dtype>* failure_blob = net_->failure_learnable_params()[net_->fc_params_ids_[i]];
      Blob<Dtype>* failure_blob = dynamic_cast<GaussianFailureMaker<Dtype>* >(fmaker_.get())->fail_iterations()[net_->fc_params_ids_[i]];
      flag_mat_p->Reshape(failure_blob->shape());
      caffe_set<Dtype>(flag_mat_p->count(), Dtype(0), flag_mat_p->mutable_cpu_data()); // initialized 0 
      GetFailFlagMat(failure_blob, flag_mat_p);
      flag_mat_vec.push_back(flag_mat_p);
    }
    
    for (int i = 1; i < size; i++) {
      // calculate zeros of neurons in fc layer i
      vector<int> zero_nums;
      Blob<Dtype>* input_flag_blob = flag_mat_vec[i-1].get();
      Blob<Dtype>* output_flag_blob = flag_mat_vec[i].get();
      int last_layer_neurons = input_flag_blob->shape()[1];
      int next_layer_neurons = output_flag_blob->shape()[0];
      const Dtype* input_flag = input_flag_blob->cpu_data();
      const Dtype* output_flag = output_flag_blob->cpu_data();
      for (int j = 0; j < input_flag_blob->shape()[0]; j++) {
	// calculate input and output zero weights number of j-th neuron in fc layer i
	Dtype s = caffe_cpu_asum<Dtype>(last_layer_neurons, &input_flag[j * last_layer_neurons]) +
	  caffe_cpu_strided_asum<Dtype>(next_layer_neurons, &output_flag[j], input_flag_blob->shape()[0]);
	//LOG(INFO) << "sum of " << i << "-th fc layer, " << j << "-th neuron";
	zero_nums.push_back(static_cast<int>(s));
      }
      // sort neuron index according to the zero nums
      // initialize original index locations
      std::vector<int> idx(zero_nums.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
	   [&zero_nums](size_t i1, size_t i2) {return zero_nums[i1] < zero_nums[i2];});
      orders.push_back(idx);
    }
  }

  template <typename Dtype>
  void RemappingFailureStrategy<Dtype>::Apply() {
    ++times_;
    if (times_ < start_ || (times_ - start_) % period_ != 0) {
      return;
    }
    int size = net_->fc_params_ids_.size();
    vector<vector<int> > orders;
    SortFCNeurons(orders); // orders size will be `size - 1`
    Blob<Dtype> remapped_weight;
    Blob<Dtype> remapped_bias;
    for (int i = 1; i < size; i++) {
      // remapping the neuron in th `i`-th fc layer
      vector<int>& order = orders[i-1];
      vector<int>& prune_order = prune_orders_[i-1];
      // rearrange the input weights
      Blob<Dtype>* input_weight_blob = net_->failure_learnable_params()[net_->fc_params_ids_[i-1]];
      int input_layer_dim = input_weight_blob->shape()[1];
      Blob<Dtype>* input_bias_blob = net_->failure_learnable_params()[net_->fc_params_ids_[i-1] + 1];
      remapped_weight.Reshape(input_weight_blob->shape());
      remapped_bias.Reshape(input_bias_blob->shape());
      caffe_copy(remapped_weight.count(), input_weight_blob->cpu_data(), remapped_weight.mutable_cpu_data());
      caffe_copy(remapped_weight.count(), input_weight_blob->cpu_diff(), remapped_weight.mutable_cpu_diff());
      caffe_copy(remapped_bias.count(), input_bias_blob->cpu_data(), remapped_bias.mutable_cpu_data());
      caffe_copy(remapped_bias.count(), input_bias_blob->cpu_diff(), remapped_bias.mutable_cpu_diff());
      for (int j = 0; j < order.size(); j++) {
	caffe_copy(input_layer_dim, remapped_weight.cpu_data() + prune_order[j] * input_layer_dim,
		   input_weight_blob->mutable_cpu_data() + order[j] * input_layer_dim);
	caffe_copy(input_layer_dim, remapped_weight.cpu_diff() + prune_order[j] * input_layer_dim,
		   input_weight_blob->mutable_cpu_diff() + order[j] * input_layer_dim);
	input_bias_blob->mutable_cpu_data()[order[j]] = remapped_weight.cpu_data()[prune_order[j]];
	input_bias_blob->mutable_cpu_diff()[order[j]] = remapped_weight.cpu_diff()[prune_order[j]];
      }
      // rearrange the output weights
      Blob<Dtype>* output_weight_blob = net_->failure_learnable_params()[net_->fc_params_ids_[i]];
      int output_layer_dim = output_weight_blob->shape()[0];
      int layer_dim = output_weight_blob->shape()[1];
      remapped_weight.Reshape(output_weight_blob->shape());
      caffe_copy(remapped_weight.count(), output_weight_blob->cpu_data(), remapped_weight.mutable_cpu_data());
      caffe_copy(remapped_weight.count(), output_weight_blob->cpu_diff(), remapped_weight.mutable_cpu_diff());
      for (int j = 0; j < order.size(); j++) {
	int from = prune_order[j];
	int to = order[j];
	for (int k = 0; k < output_layer_dim; k++) {
	  output_weight_blob->mutable_cpu_data()[k * layer_dim + to] = remapped_weight.cpu_data()[k * layer_dim + from];
	  output_weight_blob->mutable_cpu_diff()[k * layer_dim + to] = remapped_weight.cpu_diff()[k * layer_dim + from];
	}
      }
    }
  }

  template <typename Dtype>
  int GeneticFailureStrategy<Dtype>::CalculateOverallDist() {
    Dtype epsilon = 1e-20;
    const vector<Blob<Dtype>* >& failure_params = dynamic_cast<GaussianFailureMaker<Dtype>* >(fmaker_.get())->fail_iterations();
    const vector<Blob<Dtype>* >& prune_params = prune_net_->failure_learnable_params();
    int dist = 0;
    for (int i = 0; i < failure_params.size(); i++) {
      const Dtype* failure_param = failure_params[i]->cpu_data();
      const Dtype* prune_param = prune_params[i]->cpu_data();
      for (int j = 0; j < failure_params[i]->count(); j++) {
	if (prune_param[j] < epsilon && failure_param[j] < 0) {
	  // this weight cannot be prune but fail
	  dist += 1;
	}	
      }
    }
    return dist;
  }

  template <typename Dtype>
  void GeneticFailureStrategy<Dtype>::Apply() {
    ++times_;
    if (times_ < start_ || (times_ - start_) % period_ != 0) {
      return;
    }
    Dtype epsilon = 1e-20;
    int size = net_->fc_params_ids_.size();
    // calculate the cost before switching
    int before_dist = CalculateOverallDist();
    for (int i = 0; i < switch_time_; ) {
      // random select two neurons
      int layer_index = rand() % (size - 1) + 1;
      Blob<Dtype>* input_weight_blob = net_->failure_learnable_params()[net_->fc_params_ids_[layer_index-1]];
      int layer_dim = input_weight_blob->shape()[0];
      int input_layer_dim = input_weight_blob->shape()[1];
      int neuron_index1 = rand() % layer_dim;
      int neuron_index2 = rand() % layer_dim;
      if (neuron_index1 == neuron_index2) {
	// the same neuron, continue
	continue;
      }
      i++;
      // 只算weight的距离
      Blob<Dtype>* failure_blob = dynamic_cast<GaussianFailureMaker<Dtype>* >(fmaker_.get())->fail_iterations()[net_->fc_params_ids_[layer_index - 1]];
      Blob<Dtype>* input_bias_blob = net_->failure_learnable_params()[net_->fc_params_ids_[layer_index-1] + 1];
      const Dtype* input_fail_iters_p = failure_blob->cpu_data();
      //const Dtype* input_fail_values_p = failure_blob->cpu_data();
      
      Blob<Dtype>* output_weight_blob = net_->failure_learnable_params()[net_->fc_params_ids_[layer_index]];
      int output_layer_dim = output_weight_blob->shape()[0];

      // failure blob
      failure_blob = dynamic_cast<GaussianFailureMaker<Dtype>* >(fmaker_.get())->fail_iterations()[net_->fc_params_ids_[layer_index]];
      const Dtype* output_fail_iters_p = failure_blob->cpu_data();
      //const Dtype* output_fail_values_p = failure_blob->cpu_data()

      Dtype* prune_input = prune_net_->layers()[prune_net_->failure_learnable_layer_ids()[prune_net_->fc_params_ids_[layer_index-1]]]->blobs()[0]->mutable_cpu_data();
      Dtype* prune_output = prune_net_->layers()[prune_net_->failure_learnable_layer_ids()[prune_net_->fc_params_ids_[layer_index]]]->blobs()[0]->mutable_cpu_data();
      int dist_before = 0;
      int dist_after = 0;
      // calculate the gain of switching these two neurons (assume ...)
      for (int j = 0; j < input_layer_dim; j++) {
	// FIXME: not consider +-1
	if (prune_input[neuron_index1 * input_layer_dim + j] < epsilon && input_fail_iters_p[neuron_index1 * input_layer_dim + j] < 0) {
	  dist_before += 1;
	}
	if (prune_input[neuron_index2 * input_layer_dim + j] < epsilon && input_fail_iters_p[neuron_index1 * input_layer_dim + j] < 0) {
	  dist_after += 1;
	}
	if (prune_input[neuron_index2 * input_layer_dim + j] < epsilon && input_fail_iters_p[neuron_index2 * input_layer_dim + j] < 0) {
	  dist_before += 1;
	}
	if (prune_input[neuron_index1 * input_layer_dim + j] < epsilon && input_fail_iters_p[neuron_index2 * input_layer_dim + j] < 0) {
	  dist_after += 1;
	}
      }
      for (int j = 0; j < output_layer_dim; j++) {
	// FIXME: not consider +-1
	if (prune_output[j * layer_dim + neuron_index1] < epsilon && output_fail_iters_p[j * layer_dim + neuron_index1] < 0) {
	  dist_before += 1;
	}
	if (prune_output[j * layer_dim + neuron_index2] < epsilon && output_fail_iters_p[j * layer_dim + neuron_index1] < 0) {
	  dist_after += 1;
	}
	if (prune_output[j * layer_dim + neuron_index2] < epsilon && output_fail_iters_p[j * layer_dim + neuron_index2] < 0) {
	  dist_before += 1;
	}
	if (prune_output[j * layer_dim + neuron_index1] < epsilon && output_fail_iters_p[j * layer_dim + neuron_index2] < 0) {
	  dist_after += 1;
	}
      }
      Blob<Dtype> tmp_weight;
      Blob<Dtype> tmp_bias;
      // switch the neuron
      //LOG(INFO) << "dist_before: " << dist_before << "dist_after: " << dist_after;
      if (dist_after < dist_before) {
	tmp_weight.Reshape(1, 1, 1, input_layer_dim);
	// input
	caffe_copy(input_layer_dim, input_weight_blob->cpu_data() + neuron_index1 * input_layer_dim,
		   tmp_weight.mutable_cpu_data());
	caffe_copy(input_layer_dim, input_weight_blob->cpu_data() + neuron_index2 * input_layer_dim,
		   input_weight_blob->mutable_cpu_data() + neuron_index1 * input_layer_dim);
	caffe_copy(input_layer_dim, tmp_weight.cpu_data(),
		   input_weight_blob->mutable_cpu_data() + neuron_index2 * input_layer_dim);
	caffe_copy(input_layer_dim, input_weight_blob->cpu_diff() + neuron_index1 * input_layer_dim,
		   tmp_weight.mutable_cpu_diff());
	caffe_copy(input_layer_dim, input_weight_blob->cpu_diff() + neuron_index2 * input_layer_dim,
		   input_weight_blob->mutable_cpu_diff() + neuron_index1 * input_layer_dim);
	caffe_copy(input_layer_dim, tmp_weight.cpu_diff(),
		   input_weight_blob->mutable_cpu_diff() + neuron_index2 * input_layer_dim);
	
	Dtype tmp = input_bias_blob->cpu_data()[neuron_index1];
	input_bias_blob->mutable_cpu_data()[neuron_index1] = input_bias_blob->cpu_data()[neuron_index2];
	input_bias_blob->mutable_cpu_data()[neuron_index2] = tmp;
	tmp = input_bias_blob->cpu_diff()[neuron_index1];
	input_bias_blob->mutable_cpu_diff()[neuron_index1] = input_bias_blob->cpu_diff()[neuron_index2];
	input_bias_blob->mutable_cpu_diff()[neuron_index2] = tmp;

	// prune input
	caffe_copy(input_layer_dim, prune_input + neuron_index1 * input_layer_dim,
		   tmp_weight.mutable_cpu_data());
	caffe_copy(input_layer_dim, prune_input + neuron_index2 * input_layer_dim,
		   prune_input + neuron_index1 * input_layer_dim);
	caffe_copy(input_layer_dim, tmp_weight.cpu_data(),
		   prune_input + neuron_index2 * input_layer_dim);
	
	tmp = prune_input[neuron_index1];
	prune_input[neuron_index1] = prune_input[neuron_index2];
	prune_input[neuron_index2] = tmp;

	// output
	for (int k = 0; k < output_layer_dim; k++) {
	  Dtype tmp = output_weight_blob->cpu_data()[k * layer_dim + neuron_index1];
	  output_weight_blob->mutable_cpu_data()[k * layer_dim + neuron_index1] = output_weight_blob->cpu_data()[k * layer_dim + neuron_index2];
	  output_weight_blob->mutable_cpu_data()[k * layer_dim + neuron_index2] = tmp;
	  tmp = output_weight_blob->cpu_diff()[k * layer_dim + neuron_index1];
	  output_weight_blob->mutable_cpu_diff()[k * layer_dim + neuron_index1] = output_weight_blob->cpu_diff()[k * layer_dim + neuron_index2];
	  output_weight_blob->mutable_cpu_diff()[k * layer_dim + neuron_index2] = tmp;
	}
	// prune output
	for (int k = 0; k < output_layer_dim; k++) {
	  Dtype tmp = prune_output[k * layer_dim + neuron_index1];
	  prune_output[k * layer_dim + neuron_index1] = prune_output[k * layer_dim + neuron_index2];
	  prune_output[k * layer_dim + neuron_index2] = tmp;
	}
      }
    }
    int after_dist = CalculateOverallDist();
    LOG(INFO) << "dist: before: " << before_dist << " after: " << after_dist;
  }

  INSTANTIATE_CLASS(FailureStrategy);
  INSTANTIATE_CLASS(ThresholdFailureStrategy);
  INSTANTIATE_CLASS(RemappingFailureStrategy);
  INSTANTIATE_CLASS(GeneticFailureStrategy);
} // namespace caffe
