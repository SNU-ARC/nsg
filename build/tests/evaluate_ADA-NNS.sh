#!/bin/bash
export TIME=$(date '+%Y%m%d%H%M')

l_start=20
l_end=200
l_step=10

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done

l_start=250
l_end=500
l_step=50

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done

l_start=1000
l_end=2500
l_step=500

for (( l=l_start; l<=l_end; l=l+l_step )); do
	L_SIZE+=($l)
done

THREAD=(1)
K=(1 10)
TAU=(0.3)
HASH=(512)

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
  echo "Perform searching using NSG index (sift1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS sift1M/sift_base.fvecs sift1M/sift_query.fvecs sift1M.nsg ${1} ${2} sift1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    sift1M/sift_groundtruth.ivecs ${5} ${6} ${4} > sift1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform searching using NSG index (gist1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS gist1M/gist_base.fvecs gist1M/gist_query.fvecs gist1M.nsg ${1} ${2} gist1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    gist1M/gist_groundtruth.ivecs ${5} ${6} ${4} > gist1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform searching using NSG index (crawl_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS crawl/crawl_base.fvecs crawl/crawl_query.fvecs crawl.nsg ${1} ${2} crawl_nsg_result.ivecs \
    crawl/crawl_groundtruth.ivecs ${5} ${6} ${4} #> crawl_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
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
  echo "Perform searching using NSG index (deep1M_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS deep1M/deep1m_base.fvecs deep1M/deep1m_query.fvecs deep1M.nsg ${1} ${2} deep1M_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
    deep1M/deep1m_groundtruth.ivecs ${5} ${6} ${4} > deep1M_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

nsg_msong() {
  # Build a proximity graph
  if [ ! -f "msong.nsg" ]; then
    echo "Converting msong_200nn.graph kNN graph to msong.nsg"
    if [ -f "msong_200nn.graph" ]; then
      ./test_nsg_index msong/msong_base.fvecs msong_200nn.graph 40 50 500 msong.nsg > msong_index_${TIME}.log
    else
      echo "ERROR: msong_200nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (msong_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS msong/msong_base.fvecs msong/msong_query.fvecs msong.nsg ${1} ${2} msong_nsg_result_L${1}K${2}_${3}_T${4}.ivecs \
  msong/msong_groundtruth.ivecs ${5} ${6} ${4} > msong_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

nsg_glove-100() {
  # Build a proximity graph
  if [ ! -f "glove-100.nsg" ]; then
    echo "Converting glove-100_400nn.graph kNN graph to glove-100.nsg"
    if [ -f "glove-100_400nn.graph" ]; then
      ./test_nsg_index glove-100/glove-100_base.fvecs glove-100_400nn.graph 60 70 500 glove-100.nsg > glove-100_index_${TIME}.log
    else
      echo "ERROR: glove-100_400nn.graph does not exist"
      exit 1
    fi
  fi

  # Perform search
  echo "Perform searching using NSG index (glove-100_L${1}K${2}T${4}t${5}h${6})"
  sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
  ./test_nsg_optimized_search_ADA_NNS glove-100/glove-100_base.fvecs glove-100/glove-100_query.fvecs glove-100.nsg ${1} ${2} glove-100_nsg_result.ivecs \
    glove-100/glove-100_groundtruth.ivecs ${5} ${6} ${4}> glove-100_search_${3}_L${1}_K${2}_T${4}_t${5}_h${6}.log
}

if [[ ${#} -eq 1 ]]; then
  if [ "${1}" == "sift1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_sift1M ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "gist1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_gist1M ${l} ${k} ADA-NNS ${t} ${tt} 1024
          done
        done
      done
    done
  elif [ "${1}" == "crawl" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_crawl ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "deep1M" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_deep1M ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "msong" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_msong ${l} ${k} ADA-NNS ${t} ${tt} 768
          done
        done
      done
    done
  elif [ "${1}" == "glove-100" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for l_size in ${L_SIZE[@]}; do
            declare -i l=l_size
            nsg_glove-100 ${l} ${k} ADA-NNS ${t} ${tt} 512
          done
        done
      done
    done
  elif [ "${1}" == "all" ]; then
    for k in ${K[@]}; do
      for t in ${THREAD[@]}; do
        for tt in ${TAU[@]}; do
          for h in ${HASH[@]}; do
            for l_size in ${L_SIZE[@]}; do
              declare -i l=l_size
              nsg_sift1M ${l} ${k} ADA-NNS ${t} ${tt} 512
              nsg_gist1M ${l} ${k} ADA-NNS ${t} ${tt} 1024
              nsg_crawl ${l} ${k} ADA-NNS ${t} ${tt} 512
              nsg_deep1M ${l} ${k} ADA-NNS ${t} ${tt} 512
              nsg_msong ${l} ${k} ADA-NNS ${t} ${tt} 768
              nsg_glove-100 ${l} ${k} ADA-NNS ${t} ${tt} 512
            done
          done
        done
      done
    done
  else
    echo "Usage: ./evaluate_baseline.sh [dataset]"
  fi
  elif [[ ${#} -eq 6 ]]; then
    nsg_$1 $2 $3 ADA-NNS $4 $5 $6
else
  echo "Usage: ./evaluate_baseline.sh [dataset]"
fi
