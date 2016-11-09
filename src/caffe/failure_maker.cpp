#include "caffe/failure_maker.hpp"

namespace caffe {
  template <typename Dtype>
  GaussianFailureMaker<Dtype>::GaussianFailureMaker(const FailurePatternParameter& param, const shared_ptr<Net<Dtype> > net):
    FailureMaker<Dtype>(param){
    this->net_ = net;
    //fail_iterations_.resize(this->net_->failure_learnable_params().size());
    Blob<Dtype>* iters_p;
    for (int i = 0; i < this->net_->failure_learnable_params().size(); i++) {
      Blob<Dtype>* learnable_param = this->net_->failure_learnable_params()[i];
      iters_p = new Blob<Dtype>();
      iters_p->Reshape(learnable_param->shape());
      // FIXME: convert to int, guarantee nonegative
      caffe_rng_gaussian<Dtype>(learnable_param->count(), Dtype(this->param_.mean()),
				Dtype(this->param_.std()), iters_p->mutable_cpu_data());
      caffe_set<Dtype>(learnable_param->count(), Dtype(-2), iters_p->mutable_cpu_diff());
      fail_iterations_.push_back(iters_p);
    }
    //std::random_device rd;
    //gen_ = new  std::mt19937(rd());
    //d_ = new std::discrete_distribution<int>({10, 20, 10});
  }

  template <typename Dtype>
  void GaussianFailureMaker<Dtype>::Fail_cpu(int iter) {
    Dtype epsilon = 1e-19;
    for (int i = 0; i < fail_iterations_.size(); i++) {
      // remain iterations
      Dtype* iters_p = fail_iterations_[i]->mutable_cpu_data();
      // the fail value of every failed cell, use cpu_diff to store these info
      //Dtype* value_p = fail_iterations_[i]->mutable_cpu_diff();
      int count = fail_iterations_[i]->count();

      for (int j = 0; j < count; j++) {
	if (iters_p[j] <= 0) {
	  // this cell is broken
	  //this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = value_p[j];
	  this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = 0;
	} else {
	  // strategy1: not update when gradient is too small
	  if (fabs(this->net_->failure_learnable_params()[i]->cpu_diff()[j]) < epsilon) {
	    //this->net_->failure_learnable_params()[i]->mutable_cpu_diff()[j] = 0;
	    continue;
	  }
	  iters_p[j] -= 100; // batch size. FIXME: how to make this exp more general
	  if (iters_p[j] <= 0) {
	    //value_p[j] = random_collapse();
	    //this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = value_p[j];
	    this->net_->failure_learnable_params()[i]->mutable_cpu_data()[j] = 0;
	    LOG(INFO) << "failure to " << 0;//value_p[j];
	  }
	}
      }
    }
  }
  INSTANTIATE_CLASS(FailureMaker);
  INSTANTIATE_CLASS(GaussianFailureMaker);
}
