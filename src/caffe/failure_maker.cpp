#include "caffe/failure_maker.hpp"

namespace caffe {
  template <typename Dtype>
  GaussianFailureMaker<Dtype>::GaussianFailureMaker(const FailurePatternParameter& param, const shared_ptr<Net<Dtype> > net):
    FailureMaker<Dtype>(param){
    this->net_ = net;
    Blob<Dtype>* iters_p;
    std::vector<int> probs;
    if (param_.has_failure_prob()) {
      CHECK_GE(param_.failure_prob().neg(), 0) << "Probability for failure to -1 must be greater or equal than 0";
      CHECK_GE(param_.failure_prob().zero(), 0) << "Probability for failure to 0 must be greater or equal than 0";
      CHECK_GE(param_.failure_prob().pos(), 0) << "Probability for failure to 1 must be greater or equal than 0";
      probs.push_back(param_.failure_prob().neg());
      probs.push_back(param_.failure_prob().zero());
      probs.push_back(param_.failure_prob().pos());
    } else {
      probs.push_back(10);
      probs.push_back(20);
      probs.push_back(10);
    }
    int prob_sum = probs[0] + probs[1] + probs[2];
    Dtype split1 = Dtype(probs[0])/prob_sum;
    Dtype split2 = Dtype(probs[0] + probs[1])/prob_sum;
    for (int i = 0; i < this->net_->failure_learnable_params().size(); i++) {
      Blob<Dtype>* learnable_param = this->net_->failure_learnable_params()[i];
      iters_p = new Blob<Dtype>();
      iters_p->Reshape(learnable_param->shape());
      // FIXME: convert to int, guarantee nonegative
      caffe_rng_gaussian<Dtype>(learnable_param->count(), Dtype(this->param_.mean()),
				Dtype(this->param_.std()), iters_p->mutable_cpu_data());
#ifndef CPU_ONLY
      caffe_gpu_rng_uniform<Dtype>(learnable_param->count(), Dtype(0), Dtype(1),
			       iters_p->mutable_gpu_diff());
      failure_threshold<Dtype>(learnable_param->count(), iters_p->mutable_gpu_diff(), split1, split2);
#else
      caffe_rng_uniform<Dtype>(learnable_param->count(), Dtype(0), Dtype(1),
			       iters_p->mutable_cpu_diff());
      Dtype* values = iters_p->mutable_cpu_diff();
      for (int j = 0; j < learnable_param->count(); j++) {
	if (values[j] < split1) {
	  values[j] = -1;
	} else if (values[j] < split2) {
	  values[j] = 0;
	} else {
	  values[j] = 1;
	}
      }
#endif
      fail_iterations_.push_back(iters_p);
    }
  }

  template <typename Dtype>
  void GaussianFailureMaker<Dtype>::Fail_cpu(int iter) {
    Dtype epsilon = 1e-20;
    for (int i = 0; i < fail_iterations_.size(); i++) {
      // remain iterations
      Dtype* iters_p = fail_iterations_[i]->mutable_cpu_data();
      // the fail value of every failed cell, use cpu_diff to store these info
      const Dtype* value_p = fail_iterations_[i]->cpu_diff();
      int count = fail_iterations_[i]->count();
      for (int j = 0; j < count; j++) {
	if (iters_p[j] <= 0) {
	  // this cell is broken
	  this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = value_p[j];
	} else {
	  // strategy1: not update when gradient is too small
	  if (fabs(this->net_->failure_learnable_params()[i]->cpu_diff()[j]) < epsilon) {
	    //this->net_->failure_learnable_params()[i]->mutable_cpu_diff()[j] = 0;
	    continue;
	  }
	  iters_p[j] -= 100; // batch size. FIXME: how to make this exp more general
	  if (iters_p[j] <= 0) {
	    this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = value_p[j];
	    LOG(INFO) << "failure to " << value_p[j];
	  }
	}
      }
    }
  }
  INSTANTIATE_CLASS(FailureMaker);
  INSTANTIATE_CLASS(GaussianFailureMaker);
}
