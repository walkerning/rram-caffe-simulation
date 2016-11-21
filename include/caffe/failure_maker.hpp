#ifndef CAFFE_FAILURE_MAKER_HPP_
#define CAFFE_FAILURE_MAKER_HPP_

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include <iostream>
#include <random>

namespace caffe {
  template <typename Dtype>
  void failure_threshold(const int n, Dtype* values, Dtype split1, Dtype split2);

  // forward declaration
  template <typename Dtype>
  class GaussianFailureMaker;

  template <typename Dtype>
  class FailureMaker {
  public:
    explicit FailureMaker(const FailurePatternParameter& param): param_(param) {}

    static shared_ptr<FailureMaker<Dtype> > CreateMaker(const FailurePatternParameter& param, const shared_ptr<Net<Dtype> > net) {
      shared_ptr<FailureMaker<Dtype> > ptr;
      const std::string& type = param.type();
      if (type == "gaussian") {
	ptr.reset(new GaussianFailureMaker<Dtype>(param, net));
      }
      return ptr;
    }

    void Fail(int iter) {
      switch (Caffe::mode()) {
      case Caffe::CPU:
	Fail_cpu(iter);
	break;
      case Caffe::GPU:
	Fail_gpu(iter);
	int broken = 0;
	int whole = 0;
	// test how many cell is broken
	for (int i = 0; i < fail_iterations_.size(); i++) {
	  // remain iterations
	  const Dtype* iters_p = fail_iterations_[i]->cpu_data();
	  whole += fail_iterations_[i]->count();
	  // the fail value of every failed cell, use cpu_diff to store these info
	  int count = fail_iterations_[i]->count();
	  for (int j = 0; j < count; j++) {
	    if (iters_p[j] <= 0) {
	      broken += 1;
	    }
	  }
	}
	//LOG(INFO) << "broken: " << Dtype(broken)/whole;
	break;
      }
      return;
    }

    virtual void Fail_cpu(int iter)=0;
    virtual void Fail_gpu(int iter)=0;

    virtual ~FailureMaker() {}

  protected:
    FailurePatternParameter param_;
    shared_ptr<Net<Dtype> > net_;
    vector<Blob<Dtype>* > fail_iterations_;
  };

  template <typename Dtype>
  class GaussianFailureMaker: public FailureMaker<Dtype> {
  public:
    explicit GaussianFailureMaker(const FailurePatternParameter& param, const shared_ptr<Net<Dtype> > net);

#ifdef CPU_ONLY
    virtual void Fail_gpu(int iter) { NO_GPU; }
#else
    virtual void Fail_gpu(int iter);
#endif

    vector<Blob<Dtype>* >& fail_iterations() {
      return fail_iterations_;
    }

    virtual void Fail_cpu(int iter);
    virtual ~GaussianFailureMaker() {
      for (int i = 0; i < fail_iterations_.size(); i++) {
	delete(fail_iterations_[i]);
      }
    }

  private:
    using FailureMaker<Dtype>::fail_iterations_;
    using FailureMaker<Dtype>::param_;
  };

}

#endif
