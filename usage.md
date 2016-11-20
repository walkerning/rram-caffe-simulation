### 跑实验

在 `/home/foxfi/rram/caffe/examples/cifar10/gaussian_failure/` 目录下, 有一个python脚本叫作``run_gaussian_exp.py``.

这个脚本接收的参数可以通过

```
python run_gaussian_exp.py --help
```

输出如下:

```
usage: run_gaussian_exp.py [-h] [-t THRESHOLD] [-r REMAPPING] [--tag TAG]
                           [--cpu] [--prob PROB] [-y]
                           mean std device_id

positional arguments:
  mean
  std
  device_id

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
  -r REMAPPING, --remapping REMAPPING
                        <prune_order_file>[,<period>[,<start>]]
  --tag TAG             make a tag as suffix
  --cpu                 run on cpu
  --prob PROB           probability percentage for +-1 (integer: 0 ~ 100)
  -y, --yes             no ask, just yes
```

有三个必须要有的参数分别是gaussian failure的均值(mean), 标准差(std), 和这个实验在哪个gpu上运行(device_id).

其它可选option解释如下:

* `--prob PROB`: PROB是stuck at +-1的概率的百分值(是一个整数), 比如``--prob 5``的意思就是以90%的概率stuck at 0. 5%的概率stuck at +1 or -1.

* `-t THRESHOLD`: 如果不加这个参数, 则不用threshold strategy. 如果加了。 则以THRESHOLD作为diff的绝对值的分界线。小于THRESHOLD的梯度将不予更新。

* `-r REMAPPING`: 如果不加这个参数, 则不用remapping strategy. 如果加了。REMAPPING的格式是:
   ``PRUNE_ORDER_FILE[,PERIOD[,START]]``
   一共三个配置, 以逗号','分隔, 后面两个配置是可选的:
   * PRUNE_ORDER_FILE必须存在, 为作为全连接层的prune排序文件的文件名。程序将从这个文件中读入需要remapping的全连接层的permutation indexes. 一个该文件例子见`prune_order.txt`, 该文件里有两层全连接层神经元的排列顺序。
   * PERIOD: 默认为100: 为经过多少个batch(如果是100个batch, 那么就是100 * 100 = 10000个数据)进行一次remapping操作
   * START: 默认为0: 从第START个batch开始才开始每PERIOD个batch进行一次remapping strategy.

* `--tag TAG`: 这个是为了方便能够把实验结果存在不同文件名下加入的参数, TAG会被加入见后面的snapshot目录名里面。

* `--cpu`: 这个参数不用管, 是我为了用来调试, 用来在CPU上运行的参数

我们的实验一般在+-1 stuck probablity在5%的时候进行试验。所以一般都会加上``--prob 5``这个option


一个运行示例如下:

```
python run_gaussian_exp.py 100000000 30000000 2 --prob 5 
```
意思为高斯错误的均值为10^8，高斯错误标准差为3*10^7. 在2号gpu上运行(可以用``nvidia-smi``查看当前几个gpu的状态, 看看哪个gpu最空闲(被占用显存最小)). stuck at +-1 prob = 5%

----

运行实验后, snapshot的solverstate和caffemodel将存在``snapshot_{mean}_{std}{strategy}{tag}`这个目录下, 其中strategy字段如果使用了threshold strategy, 会是``threshold_<THRESHOLD>``, 如果使用了remapping strategy, 会是``remapping_<PRUNE_ORDER_FILE>``.

在该目录下还会有一个log文件, 使用`/home/foxfi/rram/caffe/examples/cifar10/plot_pic.py`这个脚本可以进行画图(画出训练的acc和loss的两张图):
```
python /home/foxfi/rram/caffe/examples/cifar10/plot_pic.py snapshot_.../log
```

可能在本机用matlab画图或者处理多个log文件在一张图上, 比较方便。此时可以用``scp``或者mobaxterm等软件带的文件拷贝功能将log拷到自己本机, 然后在本机使用`-n`参数运行该脚本, 就不会用Python的matplotlib库来画图, 而是只会打印出需要的数据, 此时可以把这个数据复制到matlab中就好了. 一个例子如下:

```
$ python plot_pic.py snapshot_5000000.0_1500000.0_var_conv_exp_prob_5
iter     accuracy    loss
0           0.1002          78.5854
500         0.1093          64.5038
1000        0.1066          71.4882
1500        0.108           71.2962
2000        0.1007          55.482
2500        0.1017          42.1225
...
```

可以看到, 第一列为训练的iteration数, 第二列为对应的test accuracy, 第三列为对应的loss


---

