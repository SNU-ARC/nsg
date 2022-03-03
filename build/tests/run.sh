#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
K=(10)
#L_SIZE=(40)
L_SIZE=(20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)

nsg_sift1M() {
  if [ ! -f "sift1M.nsg" ]; then
    echo "Converting sift1M_200nn.graph kNN graph to sift1M.nsg"
    if [ -f "sift1M_200nn.graph" ]; then
      ./test_nsg_index sift1M/sift_base.fvecs sift1M_200nn.graph 40 50 500 sift1M.nsg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift1M_200nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index (sift1M_L${l}K${2})"
  sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg ${1} ${2} sift1M_nsg_result.ivecs \
    sift1M/sift_groundtruth.ivecs 512 0.25 > sift1M_search_L${1}K${2}_${3}.log
}

nsg_gist1M() {
  if [ ! -f "gist1M.nsg" ]; then
    echo "Converting gist1M_400nn.graph kNN graph to gist1M.nsg"
    if [ -f "gist1M_400nn.graph" ]; then
      ./test_nsg_index gist1M/gist_base.fvecs gist1M_400nn.graph 60 70 500 gist1M.nsg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist1M_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index (gist1M_L${1}K${2})"
  sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg ${1} ${2} gist1M_nsg_result.ivecs \
    gist1M/gist_groundtruth.ivecs 1024 0.3 > gist1M_search_L${1}K${2}_${3}.log
}

if [ "${1}" == "sift1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      nsg_sift1M ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "gist1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      nsg_gist1M ${l} ${k} ${2}
    done
  done
elif [ "${1}" == "all" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      nsg_sift1M ${l} ${k} ${2}
      nsg_gist1M ${l} ${k} ${2}
    done
  done
else
  echo "Please use either 'sift' or 'gist' as an argument"
fi
