#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')
T=(1 2 4 8 max)
K=(1 10)
L_SIZE=(30)
#L_SIZE=(31 32 33 34 35 36 37 38 39) # sift1M 95%
#L_SIZE=(51 52 53 54 55 56 57 58 59) # crawl/deep1M 95%
#L_SIZE=(61 62 63 64 65 66 67 68 69)
#L_SIZE=(20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
#L_SIZE=(250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)

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
  echo "Perform kNN searching using NSG index (sift1M_L${l}K${2}T${4})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg ${1} ${2} sift1M_nsg_result.ivecs \
    sift1M/sift_groundtruth.ivecs 512 0.25 ${4}> sift1M_search_L${1}K${2}_${3}_T${4}.log
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
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg ${1} ${2} gist1M_nsg_result.ivecs \
    gist1M/gist_groundtruth.ivecs 1024 0.3 ${4}> gist1M_search_L${1}K${2}_${3}_T${4}.log
}

nsg_deep1M() {
  if [ ! -f "deep1M.nsg" ]; then
    echo "Converting deep1M_400nn.graph kNN graph to deep1M.nsg"
    if [ -f "deep1M_400nn.graph" ]; then
      ./test_nsg_index deep1M/deep1m_base.fvecs deep1M_400nn.graph 200 40 1000 deep1M.nsg > deep1M_index_${TIME}.log
    else
      echo "ERROR: deep1M_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index (deep1M_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.nsg ${1} ${2} deep1M_nsg_result.ivecs \
    deep1M/deep1m_groundtruth.ivecs 512 0.3 ${4}> deep1M_search_L${1}K${2}_${3}_T${4}.log
}

nsg_glove-100() {
  if [ ! -f "glove-100.nsg" ]; then
    echo "Converting glove-100_400nn.graph kNN graph to glove-100.nsg"
    if [ -f "glove-100_400nn.graph" ]; then
      ./test_nsg_index glove-100/glove-100_base.fvecs glove-100_400nn.graph 150 50 1000 glove-100.nsg > glove-100_index_${TIME}.log
    else
      echo "ERROR: glove-100_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index (glove-100_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.nsg ${1} ${2} glove-100_nsg_result.ivecs \
    glove-100/glove-100_groundtruth.ivecs 512 0.3 ${4}> glove-100_search_L${1}K${2}_${3}_T${4}.log
}

nsg_crawl() {
  if [ ! -f "crawl.nsg" ]; then
    echo "Converting crawl_400nn.graph kNN graph to crawl.nsg"
    if [ -f "crawl_400nn.graph" ]; then
      ./test_nsg_index crawl/crawl_base.fvecs crawl_400nn.graph 150 50 1000 crawl.nsg > crawl_index_${TIME}.log
    else
      echo "ERROR: crawl_400nn.graph does not exist"
      exit 1
    fi
  fi
  echo "Perform kNN searching using NSG index (crawl_L${1}K${2})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.nsg ${1} ${2} crawl_nsg_result.ivecs \
    crawl/crawl_groundtruth.ivecs 512 0.3 ${4}> crawl_search_L${1}K${2}_${3}_T${4}.log
}

if [ "${1}" == "sift1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_sift1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "gist1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_gist1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "deep1M" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_deep1M ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "glove-100" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_glove-100 ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "crawl" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_crawl ${l} ${k} ${2} ${t}
      done
    done
  done
elif [ "${1}" == "all" ]; then
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=l_size
      for t in ${T[@]}; do
        nsg_sift1M ${l} ${k} ${2} ${t}
        nsg_gist1M ${l} ${k} ${2} ${t}
        nsg_deep1M ${l} ${k} ${2} ${t}
#        nsg_glove-100 ${l} ${k} ${2} ${t}
        nsg_crawl ${l} ${k} ${2} ${t}
      done
    done
  done
else
  echo "Please use either 'sift1M' or 'gist1M' or 'deep1M' or 'glove-100' or 'crawl' or 'all' as an argument"
fi
