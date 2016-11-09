#ifndef CAFFE_FAILURE_MAKER_HPP_
#define CAFFE_FAILURE_MAKER_HPP_

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include <iostream>
#include <random>

namespace caffe {
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

    virtual void Fail_cpu(int iter);
    virtual ~GaussianFailureMaker() {
      delete(gen_);
      delete(d_);
      for (int i = 0; i < fail_iterations_.size(); i++) {
	delete(fail_iterations_[i]);
      }
    }

  protected:
    inline int random_collapse() {
      //return (*d_)(*gen_) - 1;
      return 0;
    }

  private:
    std::mt19937* gen_;
    std::discrete_distribution<int>* d_;
    vector<Blob<Dtype>* > fail_iterations_;
  };
}

#endif
