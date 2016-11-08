#ifndef CAFFE_FAILURE_STRATEGY_HPP_
#define CAFFE_FAILURE_STRATEGY_HPP_

#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "caffe/failure_maker.hpp"
#include <iostream>
#include <random>

namespace caffe {
  // forward declaration
  template <typename Dtype>
  class ThresholdFailureStrategy;

  template <typename Dtype>
  class FailureStrategy {
  public:
    explicit FailureStrategy(const FailureStrategyParameter& param, const shared_ptr<FailureMaker<Dtype> > fmaker,
			     const shared_ptr<Net<Dtype> > net): param_(param), fmaker_(fmaker), net_(net) {}

    virtual inline const char* type() const { return ""; }
    static shared_ptr<FailureStrategy<Dtype> > CreateStrategy(const FailureStrategyParameter& param, 
							      const shared_ptr<FailureMaker<Dtype> > fmaker,
							      const shared_ptr<Net<Dtype> > net) {
      shared_ptr<FailureStrategy<Dtype> > ptr;
      const std::string& type = param.type();
      if (type == "threshold") {
	ptr.reset(new ThresholdFailureStrategy<Dtype>(param, fmaker, net));
      }
      return ptr;
    }

    virtual void Apply()=0;

    virtual ~FailureStrategy() {}

  protected:
    FailureStrategyParameter param_;
    shared_ptr<FailureMaker<Dtype> > fmaker_;
    shared_ptr<Net<Dtype> > net_;
  };

  template <typename Dtype>
  class ThresholdFailureStrategy: public FailureStrategy<Dtype> {
  public:
    explicit ThresholdFailureStrategy(const FailureStrategyParameter& param, 
				      const shared_ptr<FailureMaker<Dtype> > fmaker,
				      const shared_ptr<Net<Dtype> > net): FailureStrategy<Dtype>(param, fmaker, net) {
      threshold_ = this->param_.threshold();
      CHECK_GT(threshold_, 0) << "`threshold` must be postive!";
    }

    virtual inline const char* type() const { return "threshold"; }
    virtual void Apply();
    virtual ~ThresholdFailureStrategy() {}
    using FailureStrategy<Dtype>::net_;
    using FailureStrategy<Dtype>::fmaker_;
    using FailureStrategy<Dtype>::param_;

  private:
    Dtype threshold_;
  };
}

#endif
