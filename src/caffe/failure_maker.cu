#include "caffe/util/device_alternate.hpp"
#include "caffe/failure_maker.hpp"

namespace caffe {
  template <typename Dtype>
  __global__ void FailureThresholdKernel(const int n, Dtype* values, Dtype split1, Dtype split2) {
    CUDA_KERNEL_LOOP(index, n) {
      if (values[index] < split1) {
	values[index] = -1;
      } else if (values[index] < split2) {
	values[index] = 0;
      } else {
	values[index] = 1;
      }
    }
  }

  template <typename Dtype>
  void failure_threshold(const int n, Dtype* values, Dtype split1, Dtype split2) {
    FailureThresholdKernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, values, split1, split2);
  }

  template <typename Dtype>
  __global__ void FailKernel(const int n, Dtype* iters, const Dtype* values, Dtype* data, const Dtype* diff) {
    Dtype epsilon = 1e-20;
    CUDA_KERNEL_LOOP(index, n) {
      if (iters[index] <= 0) {
	// this cell is already broken
	data[index] = values[index];
      } else {
	// strategy1: not update when gradient is too small
	if (diff[index] < epsilon && diff[index] > -epsilon) {
	  continue;
	}
	iters[index] -= 100; // batch size. FIXME: how to make this exp more general
	if (iters[index] <= 0) {
	  data[index] = values[index];
	}
      }
    }
  }

  template <typename Dtype>
  void GaussianFailureMaker<Dtype>::Fail_gpu(int iter) {
    for (int i = 0; i < fail_iterations_.size(); i++) {
      int count = fail_iterations_[i]->count();
      int N = CAFFE_GET_BLOCKS(count);
      // curandState_t* states;
      // cudaMalloc((void**) &states, N * sizeof(curandState_t));
      // InitRandom<<<N, 1>>>(time(NULL), states);

      FailKernel<Dtype><<<N, CAFFE_CUDA_NUM_THREADS>>>(count, 
						       fail_iterations_[i]->mutable_gpu_data(),
						       fail_iterations_[i]->mutable_gpu_diff(),
						       this->net_->failure_learnable_params()[i]->mutable_gpu_data(),
						       this->net_->failure_learnable_params()[i]->gpu_diff());
	}
  }
  
  template void GaussianFailureMaker<double>::Fail_gpu(int);
  template void GaussianFailureMaker<float>::Fail_gpu(int);

  template void failure_threshold<float>(const int n, float* values, float split1, float split2);
  template void failure_threshold<double>(const int n, double* values, double split1, double split2);
}
