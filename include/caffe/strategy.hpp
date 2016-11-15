#ifndef CAFFE_FAILURE_STRATEGY_HPP_
#define CAFFE_FAILURE_STRATEGY_HPP_

#include <cmath>
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "caffe/failure_maker.hpp"
#include <iostream>
#include <random>

namespace caffe {
  // forward declaration
  template <typename Dtype>
  class Solver;

  template <typename Dtype>
  class ThresholdFailureStrategy;

  template <typename Dtype>
  class FailureStrategy {
  public:
    explicit FailureStrategy(const FailureStrategyParameter& param, const shared_ptr<FailureMaker<Dtype> > fmaker,
			     const shared_ptr<Net<Dtype> > net, const Solver<Dtype>* solver): param_(param), fmaker_(fmaker), net_(net), solver_(solver) {}

    virtual inline const char* type() const { return ""; }
    static shared_ptr<FailureStrategy<Dtype> > CreateStrategy(const FailureStrategyParameter& param, 
							      const shared_ptr<FailureMaker<Dtype> > fmaker,
							      const shared_ptr<Net<Dtype> > net,
							      const Solver<Dtype>* solver) {
      shared_ptr<FailureStrategy<Dtype> > ptr;
      const std::string& type = param.type();
      if (type == "threshold") {
	ptr.reset(new ThresholdFailureStrategy<Dtype>(param, fmaker, net, solver));
      } else if (type == "remapping") {
	ptr.reset(new RemappingFailureStrategy<Dtype>(param, fmaker, net, solver));
      }
      return ptr;
    }

    virtual void Apply()=0;

    virtual ~FailureStrategy() {}

  protected:
    FailureStrategyParameter param_;
    shared_ptr<FailureMaker<Dtype> > fmaker_;
    shared_ptr<Net<Dtype> > net_;
    const Solver<Dtype>* solver_;
  };

  template <typename Dtype>
  class ThresholdFailureStrategy: public FailureStrategy<Dtype> {
  public:
    explicit ThresholdFailureStrategy(const FailureStrategyParameter& param, 
				      const shared_ptr<FailureMaker<Dtype> > fmaker,
				      const shared_ptr<Net<Dtype> > net,
				      const Solver<Dtype>* solver): FailureStrategy<Dtype>(param, fmaker, net, solver) {
      threshold_ = this->param_.threshold();
      CHECK_GT(threshold_, 0) << "`threshold` must be postive!";
    }

    virtual inline const char* type() const { return "threshold"; }
    virtual void Apply();
    virtual ~ThresholdFailureStrategy() {}
    using FailureStrategy<Dtype>::net_;
    using FailureStrategy<Dtype>::fmaker_;
    using FailureStrategy<Dtype>::param_;
    using FailureStrategy<Dtype>::solver_;

  private:
    Dtype threshold_;
  };

  template <typename Dtype>
  class RemappingFailureStrategy: public FailureStrategy<Dtype> {
  public:
    explicit RemappingFailureStrategy(const FailureStrategyParameter& param, 
				      const shared_ptr<FailureMaker<Dtype> > fmaker,
				      const shared_ptr<Net<Dtype> > net,
				      const Solver<Dtype>* solver): FailureStrategy<Dtype>(param, fmaker, net, solver) {
      // sort each neurons according to its output/input prunes?
      
    }

    virtual inline const char* type() const { return "remapping"; }
    virtual void Apply();
    virtual ~RemappingFailureStrategy() {}
    using FailureStrategy<Dtype>::net_;
    using FailureStrategy<Dtype>::fmaker_;
    using FailureStrategy<Dtype>::param_;
    using FailureStrategy<Dtype>::solver_;
  };
}

#endif
