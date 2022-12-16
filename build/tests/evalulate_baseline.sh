#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
MAX_THREADS=`nproc --all`
THREAD=(1)
K=(10)
L_SIZE=(54)

nsg_sift1M() {
  # Build a proximity graph
  if [ ! -f "sift1M.nsg" ]; then
    echo "Converting sift1M_200nn.graph kNN graph to sift1M.nsg"
    if [ -f "sift1M_200nn.graph" ]; then
      ./test_nsg_index sift1M/sift_base.fvecs sift1M_200nn.graph 40 50 500 sift1M.nsg > sift1M_index_${TIME}.log
    else
      echo "ERROR: sift1M_200nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (sift1M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg ${1} ${2} sift1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift1M/sift_groundtruth.ivecs ${4}> sift1M_search_L${1}K${2}_${3}_T${4}.log
}

nsg_gist1M() {
  # Build a proximity graph
  if [ ! -f "gist1M.nsg" ]; then
    echo "Converting gist1M_400nn.graph kNN graph to gist1M.nsg"
    if [ -f "gist1M_400nn.graph" ]; then
      ./test_nsg_index gist1M/gist_base.fvecs gist1M_400nn.graph 60 70 500 gist1M.nsg > gist1M_index_${TIME}.log
    else
      echo "ERROR: gist1M_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (gist1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg ${1} ${2} gist1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    gist1M/gist_groundtruth.ivecs ${4}> gist1M_search_L${1}K${2}_${3}_T${4}.log
}

nsg_deep1M() {
  # Build a proximity graph
  if [ ! -f "deep1M.nsg" ]; then
    echo "Converting deep1M_400nn.graph kNN graph to deep1M.nsg"
    if [ -f "deep1M_400nn.graph" ]; then
      ./test_nsg_index deep1M/deep1m_base.fvecs deep1M_400nn.graph 200 40 1000 deep1M.nsg > deep1M_index_${TIME}.log
    else
      echo "ERROR: deep1M_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (deep1M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.nsg ${1} ${2} deep1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep1M/deep1m_groundtruth.ivecs ${4}> deep1M_search_L${1}K${2}_${3}_T${4}.log
}

nsg_crawl() {
  # Build a proximity graph
  if [ ! -f "crawl.nsg" ]; then
    echo "Converting crawl_400nn.graph kNN graph to crawl.nsg"
    if [ -f "crawl_400nn.graph" ]; then
      ./test_nsg_index crawl/crawl_base.fvecs crawl_400nn.graph 150 50 1000 crawl.nsg > crawl_index_${TIME}.log
    else
      echo "ERROR: crawl_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (crawl_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.nsg ${1} ${2} crawl_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    crawl/crawl_groundtruth.ivecs ${4}> crawl_search_L${1}K${2}_${3}_T${4}.log
}

nsg_deep100M_16T() {
  export sub_num=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
  for id in ${sub_num[@]}; do
    if [ ! -f "deep100M_${id}.nsg" ]; then
      echo "Converting deep100M_400nn_${id}.graph kNN graph to deep100M_${id}.nsg"
      if [ -f "efanna_graph/deep100M_400nn_${id}.graph" ]; then
        ./test_nsg_index deep100M/deep100M_base_${id}.fvecs efanna_graph/deep100M_400nn_${id}.graph 200 40 1000 deep100M_${id}.nsg > deep100M_index_${id}_${TIME}.log
      else
        echo "ERROR: deep100M_400nn_${id}.graph does not exist"
        exit 1
      fi
    fi
  done
  echo "Perform kNN searching using NSG index (deep100M_L${1}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_multi deep100M/deep100M_base.fvecs deep100M/deep100M_query.fvecs deep100M.nsg ${1} ${2} \
    deep100M_nsg_result.ivecs deep100M/deep100M_groundtruth.ivecs 512 0.3 ${4} > deep100M_search_L${1}K${2}_${3}_T${4}.log
}

if [[ ${#} -eq 1 ]]; then
  if [ "${1}" == "sift1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_sift1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "gist1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_gist1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "crawl" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_crawl ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "deep1M" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_deep1M ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "deep100M_16T" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_deep100M_16T ${l} ${k} baseline ${t}
        done
      done
    done
  elif [ "${1}" == "all" ]; then
    for k in ${K[@]}; do
      for l_size in ${L_SIZE[@]}; do
        declare -i l=l_size
        for t in ${THREAD[@]}; do
          nsg_sift1M ${l} ${k} baseline ${t}
          nsg_gist1M ${l} ${k} baseline ${t}
          nsg_crawl ${l} ${k} baseline ${t}
          nsg_deep1M ${l} ${k} baseline ${t}
        done
      done
    done
  else
    echo "Usage: ./evaluate_baseline.sh [dataset]"
  fi
else
  echo "Usage: ./evaluate_baseline.sh [dataset]"
fi
