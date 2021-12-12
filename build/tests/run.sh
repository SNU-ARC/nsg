#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
if [ "${1}" == "sift1M" ]; then
  if [ ! -f "sift1M.nsg" ]; then
    echo "Converting sift_200nn.graph kNN graph to sift.nsg"
    if [ -f "sift_200nn.graph" ]; then
      ./test_nsg_index sift1M/sift_base.fvecs sift_200nn.graph 40 50 500 sift1M.nsg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift_200nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index"
  ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg 20 10 sift1M_nsg_result.ivecs sift1M/sift_groundtruth.ivecs > sift1M_search_${TIME}.log
elif [ "${1}" == "gist1M" ]; then
  if [ ! -f "gist1M.nsg" ]; then
    echo "Converting gist_400nn.graph kNN graph to gist.nsg"
    if [ -f "gist_400nn.graph" ]; then
      ./test_nsg_index gist1M/gist_base.fvecs gist_400nn.graph 60 70 500 gist1M.nsg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index"
  ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg 10 10 gist1M_nsg_result.ivecs gist1M/gist_groundtruth.ivecs > gist1M_search_${TIME}.log
elif [ "${1}" == "all" ]; then
  if [ ! -f "sift1M.nsg" ]; then
    echo "Converting sift_200nn.graph kNN graph to sift.nsg"
    if [ -f "sift_200nn.graph" ]; then
      ./test_nsg_index sift1M/sift_base.fvecs sift_200nn.graph 40 50 500 sift1M.nsg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift_200nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index"
  ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg 10 10 sift1M_nsg_result.ivecs > sift1M_search_${TIME}.log

  if [ ! -f "gist1M.nsg" ]; then
    echo "Converting gist_400nn.graph kNN graph to gist.nsg"
    if [ -f "gist_400nn.graph" ]; then
      ./test_nsg_index gist1M/gist_base.fvecs gist_400nn.graph 60 70 500 gist1M.nsg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index"
  ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg 10 10 gist1M_nsg_result.ivecs > gist1M_search_${TIME}.log
else
  echo "Please use either 'sift' or 'gist' as an argument"
fi
