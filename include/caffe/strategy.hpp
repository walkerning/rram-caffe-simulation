#ifndef CAFFE_FAILURE_STRATEGY_HPP_
#define CAFFE_FAILURE_STRATEGY_HPP_

#include <cmath>
#include <fstream>
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "caffe/failure_maker.hpp"
#include <iostream>
#include <random>
#include <algorithm>

namespace caffe {
  // forward declaration
  template <typename Dtype>
  class Solver;

  template <typename Dtype>
  class ThresholdFailureStrategy;

  template <typename Dtype>
  class RemappingFailureStrategy;

  template <typename Dtype>
  class GeneticFailureStrategy;

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
      } else if (type == "genetic") {
	ptr.reset(new GeneticFailureStrategy<Dtype>(param, fmaker, net, solver));
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
      // 每period_个batch读出数据做一次pruning, 通过行列permutation尽量把已经坏了的元素换到可以pruning的元素上.
      // 如果这个period 太小, 可能会出现来回震荡的效果
      // 而且要一开始先训一下再说... 不然不一定可以prune...
      period_ = param_.period();
      start_ = param_.start();
      //prune_ratio_ = param_.prune_ratio();
      CHECK_GT(period_, 0) << "`period` must be postive!";
      CHECK_GE(start_, 0) << "`start` must be non-negative!";
      times_ = 0;
      // 将软件上pretrained的网络进行prune(这个可以在外面做。。用python容易多了), 然后把这些的神经元按照prune的入度或者出度. 或者和前一层/后一层的联系...排序. 得到一个对于神经元的序。(p1, p2, p3 ... pn)
      // 在每period个batch后. 对于这几层全连接神经元. 对坏链接做类似的排序得到(q1, q2, q3, ... qn). 然后将第q1个(rram上的实际硬件上的神经元概念)神经元用来代表p1. 认为这个网络可以模拟原网络. 所以把p1对应的那一列的输入输出weight写到q1里
      // 从文件中读出prune_order_
      CHECK(param.has_prune_order_file()) << "remapping failure strategy must have a prune order file.";
      std::ifstream fs(param.prune_order_file());
      if (!fs.is_open()) {
	LOG(FATAL) << "打开文件 " << param.prune_order_file() << " 失败";
      }
      for (int i = 1; i < net_->fc_params_ids_.size(); i++) {
	vector<int> prune_order;
	int num_neurons = net_->failure_learnable_params()[net_->fc_params_ids_[i]]->shape()[1];
	for (int j = 0; j < num_neurons; j++) {
	  int next_num;
	  fs >> std::ws;
	  if ((next_num = fs.peek()) == EOF) {
	    LOG(FATAL) << "prune order file not correct";
	  }
	  fs >> next_num;
	  fs >> std::ws;
	  prune_order.push_back(next_num);
	}
	prune_orders_.push_back(prune_order);
      }
    }

    virtual inline const char* type() const { return "remapping"; }
    virtual void Apply();
    virtual ~RemappingFailureStrategy() {}
    void SortFCNeurons(vector<vector<int> >& orders);
    void GetFailFlagMat(Blob<Dtype>* failure_blob, shared_ptr<Blob<Dtype> > flag_mat_p);
    using FailureStrategy<Dtype>::net_;
    using FailureStrategy<Dtype>::fmaker_;
    using FailureStrategy<Dtype>::param_;
    using FailureStrategy<Dtype>::solver_;

  private:
    int period_;
    int start_;
    int times_;
    //float prune_ratio_;
    vector<vector<int> > prune_orders_;
  };

  template <typename Dtype>
  class GeneticFailureStrategy: public FailureStrategy<Dtype> {
  public:
    explicit GeneticFailureStrategy(const FailureStrategyParameter& param, 
				    const shared_ptr<FailureMaker<Dtype> > fmaker,
				    const shared_ptr<Net<Dtype> > net,
				    const Solver<Dtype>* solver): FailureStrategy<Dtype>(param, fmaker, net, solver) {
      switch_time_ = param.switch_time();
      CHECK_GT(switch_time_, 0) << "`switch_time_` must be postive!";
      period_ = param_.period();
      start_ = param_.start();
      //prune_ratio_ = param_.prune_ratio();
      CHECK_GT(period_, 0) << "`period` must be postive!";
      CHECK_GE(start_, 0) << "`start` must be non-negative!";
      times_ = 0;
      
      // read the prune_net prototxt
      CHECK(param.has_prune_net_file()) << "genetic failure strategy must have a prune net file.";
      CHECK(param.has_prune_model_file()) << "genetic failure strategy must have a prune model file.";
      prune_net_ = new Net<Dtype>(param.prune_net_file(), caffe::TEST);
      prune_net_->CopyTrainedLayersFrom(param.prune_model_file());
    }

    virtual inline const char* type() const { return "genetic"; }
    virtual void Apply();
    virtual ~GeneticFailureStrategy() {}
    using FailureStrategy<Dtype>::net_;
    using FailureStrategy<Dtype>::fmaker_;
    using FailureStrategy<Dtype>::param_;
    using FailureStrategy<Dtype>::solver_;

  private:
    int switch_time_;
    int period_;
    int start_;
    int times_;
    Net<Dtype>* prune_net_;
  };

}

#endif
