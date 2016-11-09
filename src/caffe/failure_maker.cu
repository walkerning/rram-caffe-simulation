#include "caffe/util/device_alternate.hpp"
#include "caffe/failure_maker.hpp"

namespace caffe {
  template <typename Dtype>
  __global__ void FailKernel(const int n, Dtype* iters, Dtype* data, const Dtype* diff) {//Dtype* values
    Dtype epsilon = 1e-20;
    CUDA_KERNEL_LOOP(index, n) {
      if (iters[index] <= 0) {
	// this cell is already broken
	//data[index] = values[index];
	data[index] = 0;//values[index];
      } else {
	// strategy1: not update when gradient is too small
	if (diff[index] < epsilon && diff[index] > -epsilon) {
	  continue;
	}
	iters[index] -= 100; // batch size. FIXME: how to make this exp more general
	if (iters[index] <= 0) {
	  //values[index] = 0;
	  data[index] = 0;
	}
      }
    }
  }

  template <typename Dtype>
  void GaussianFailureMaker<Dtype>::Fail_gpu(int iter) {
    for (int i = 0; i < fail_iterations_.size(); i++) {
      int count = fail_iterations_[i]->count();
      FailKernel<Dtype><<<CAFFE_GET_BLOCKS(count),
	CAFFE_CUDA_NUM_THREADS>>>(count, 
				  fail_iterations_[i]->mutable_gpu_data(),
				  //fail_iterations_[i]->mutable_gpu_diff();
				  this->net_->failure_learnable_params()[i]->mutable_gpu_data(),
				  this->net_->failure_learnable_params()[i]->gpu_diff());
    }
  }
  
  template void GaussianFailureMaker<double>::Fail_gpu(int);
  template void GaussianFailureMaker<float>::Fail_gpu(int);
}
