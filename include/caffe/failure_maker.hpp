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

    virtual void Fail(int iter) {}

    virtual ~FailureMaker() {}

  protected:
    FailurePatternParameter param_;
    shared_ptr<Net<Dtype> > net_;
  };

  template <typename Dtype>
  class GaussianFailureMaker: public FailureMaker<Dtype> {
  public:
    explicit GaussianFailureMaker(const FailurePatternParameter& param, const shared_ptr<Net<Dtype> > net);

    virtual void Fail(int iter);

  protected:
    inline int random_collapse() {
      return (*d_)(*gen_) - 1;
    }

  private:
    std::mt19937* gen_;
    std::discrete_distribution<int>* d_;
    vector<Blob<Dtype>* > fail_iterations_;
  };
}

#endif
