#!/bin/bash
cd .. && cmake -DCMAKE_GUILD_TYPE=Releas .. && make -j && cd tests

export TIME=$(date '+%m%d%H%M')
K=(10)
#L_SIZE=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 )
#L_SIZE=( 20 21 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 )
dataset=( "sift1M" "gist1M" "glove-100" "crawl" "deep1M" )
basename=( "sift" "gist" "glove-100" "crawl" "deep1m" )
efanna=( "sift_200nn.graph" "gist_400nn.graph" "glove_400nn.knng" "crawl_400nn.knng" "deep_400nn.knng" )
L=( 40 60 150 50 200 )
R=( 50 70 50 40 40 )
C=( 500 500 1000 1000 1000 )

run() {
  data=${dataset[$1]}
  efanna_graph=${efanna[$1]}
  base=${basename[$1]}

  echo "@ Running ${data}"

  if [ ! -f "${data}.nsg" ]; then
    echo "Converting ${efanna_graph} kNN graph to ${data}.nsg"
    if [ -f "${efanna_graph}" ]; then
      ./test_nsg_index ${data}/${base}_base.fvecs ${efanna_graph} ${L[$1]} ${R[$1]} ${C[$1]} ${data}.nsg > ${data}_index_${TIME}.log
    else
      echo "ERROR: ${efanna_graph} does not exist"
      exit 1
    fi
  fi
  for k in ${K[@]}; do
    for l_size in ${L_SIZE[@]}; do
      declare -i l=k*l_size
      echo "@@ Perform kNN searching using NSG index (L${l}K${k})" 
      echo "L${l}" >> ${data}_baseline.log
      echo "L${l}" >> ${data}_exact_sort.log
      echo "L${l}" >> ${data}_exact_aid.log
      echo "L${l}" >> ${data}_ours.log
      echo "************ baseline ************" #>> ${data}_baseline.log
      ./baseline ${data}/${base}_base.fvecs ${data}/${base}_query.fvecs ${data}.nsg $l $k ${data}_nsg_result.ivecs ${data}/${base}_groundtruth.ivecs >> ${data}_baseline.log
      echo "------------- exact sort --------------" #>> ${data}_exact_sort.log
      ./exact_sort ${data}/${base}_base.fvecs ${data}/${base}_query.fvecs ${data}.nsg $l $k ${data}_nsg_result.ivecs ${data}/${base}_groundtruth.ivecs >> ${data}_exact_sort.log
      echo "------------- exact aid --------------" #>> ${data}_exact_aid.log
      ./exact_aid ${data}/${base}_base.fvecs ${data}/${base}_query.fvecs ${data}.nsg $l $k ${data}_nsg_result.ivecs ${data}/${base}_groundtruth.ivecs >> ${data}_exact_aid.log
      echo "============== ours ==============" #>> ${data}_ours.log
      ./ours ${data}/${base}_base.fvecs ${data}/${base}_query.fvecs ${data}.nsg $l $k ${data}_nsg_result.ivecs ${data}/${base}_groundtruth.ivecs >> ${data}_ours.log
    done
  done
}

if [ "$1" == "all" ]; then
  for i in 0 1 2 3 4; do
      run $i $2
  done
elif [ "$1" == "sift1M" ]; then
  run 0 $2
elif [ "$1" == "gist1M" ]; then
  run 1 $2
elif [ "$1" == "glove-100" ]; then
  run 2 $2
elif [ "$1" == "crawl" ]; then
  run 3 $2
elif [ "$1" == "deep1M" ]; then
  run 4 $2
else
  echo "Error: Check data name"
fi
