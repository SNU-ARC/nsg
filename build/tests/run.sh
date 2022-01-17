#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
#K=(10)
#L_SIZE=(140)
#K=(10)
#L_SIZE=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
K=(100)
L_SIZE=(200)
#L_SIZE=(100 110 120 130 140 150 160 170 180 190 200)

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
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
#      declare -i l=k*l_size
      declare -i l=l_size
      echo "Perform kNN searching using NSG index (L${l}K${k})"
      ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg $l $k sift1M_nsg_result.ivecs \
      sift1M/sift_groundtruth.ivecs > sift1M_search_L${l}K${k}_${2}.log
    done
  done
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
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=k*l_size
      echo "Perform kNN searching using NSG index (L${l}K${k})"
      ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg $l $k gist1M_nsg_result.ivecs \
      gist1M/gist_groundtruth.ivecs > gist1M_search_L${l}K${k}_${2}.log
    done
  done
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
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=k*l_size
      echo "Perform kNN searching using NSG index (L${l}K${k})"
      ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg $l $k sift1M_nsg_result.ivecs \
      sift1M/sift_groundtruth.ivecs > sift1M_search_L${l}K${k}_${2}.log
    done
  done

  if [ ! -f "gist1M.nsg" ]; then
    echo "Converting gist_400nn.graph kNN graph to gist.nsg"
    if [ -f "gist_400nn.graph" ]; then
      ./test_nsg_index gist1M/gist_base.fvecs gist_400nn.graph 60 70 500 gist1M.nsg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist_400nn.graph does not exist"
      exit 1
    fi
  fi
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=k*l_size
      echo "Perform kNN searching using NSG index (L${l}K${k})"
      ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg $l $k gist1M_nsg_result.ivecs \
      gist1M/gist_groundtruth.ivecs > gist1M_search_L${l}K${k}_${2}.log
    done
  done
else
  echo "Please use either 'sift' or 'gist' as an argument"
fi
