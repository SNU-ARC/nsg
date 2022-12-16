# NSG 

This repository is NSG with greedy search method (baseline) and ADA-NNS.

Please refer to original [readme](https://github.com/SNU-ARC/nsg/blob/master/README.md).

## Building Instruction

### Prerequisites

+ GCC 4.9+ with OpenMP
+ CMake 2.8+
+ Boost 1.55+
+ [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)

**IMPORTANT NOTE: this code uses AVX-256 intructions for fast distance computation, so your machine MUST support AVX-256 intructions, this can be checked using `cat /proc/cpuinfo | grep avx2`.** 

### Compile On Ubuntu/Debian

1. Install Dependencies:

```shell
$ sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev
```

2. Compile NSG:

```shell
$ git clone https://github.com/SNU-ARC/nsg
$ cd nsg/
$ git checkout graphANNS
$ cd build/
$ ./build.sh
```

## Usage

We provide the script which can build and search for in memory-resident indices. The scripts locate under directory `tests/.` For the description of original main binaries, please refer to original [readme](https://github.com/SNU-ARC/nsg/blob/master/README.md).

### Building NSG Index

To use NSG for ANNS, an NSG index must be built first. Here are the instructions for building NSG.

#### Step 1. Build kNN Graph

Firstly, we need to prepare a kNN graph.

NSG suggests to use [efanna\_graph](https://github.com/ZJULearning/efanna\_graph) to build this kNN graph. We provide the script to build kNN graphs for various datasets. Please refer to [efanna\_graph](https://github.com/SNU-ARC/efanna\_graph) and checkout `graphANNS` branch.

You can also use any alternatives, such as [Faiss](https://github.com/facebookresearch/faiss).

#### Step 2. Build NSG index and search via NSG index

Secondly, we will convert the kNN graph to our NSG index and perform search.

To use the greedy search, use the `tests/evaluate_baseline.sh` script:
```shell
$ cd tests/
$ ./evaluate_baseline.sh [dataset] [log_suffix]
```
The argument is as follows:

(i) dataset: Name of the dataset. The script supports various real datasets (e.g., SIFT1M, GIST1M, CRAWL, DEEP1M, DEEP100M_16T).

(ii) log\_suffix: We print the result as the log. The log will be '[dataset]\_search\_L[L]K[K]\_[log\_suffix]\_T[num\_threads].log'.

To change parameter for search (e.g., K, L, number of threads), open `evaluate_baseline.sh` and modify the parameter `K, L_SIZE, T`.

To use the ADA-NNS, use the `tests/evaluate_ADA-NNS.sh` script:
```shell
$ cd tests/
$ ./evaluate_ADA-NNS.sh [dataset] [log_suffix]
```
The arguments are same as above in `evaluate_baseline.sh`.

To change parameter for search (e.g., K, L, number of threads), open `evaluate_ADA-NNS.sh` and modify the parameter `K, L_SIZE, T`.
