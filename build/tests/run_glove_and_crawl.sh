#!/bin/bash
cd .. && cmake -DCMAKE_GUILD_TYPE=Releas .. && make -j && cd tests

dataset="glove-100"
base="glove-100/glove-100_base.fvecs"
query="glove-100/glove-100_query.fvecs"
gt="glove-100/glove-100_groundtruth.ivecs"
K=10
L_SIZE=100
efanna="glove_400nn.knng"
L=( 150 )
R=( 50 )
C=( 1000 )

for l in ${L[@]}; do
  for r in ${R[@]}; do
    for c in ${C[@]}; do
      echo "============ ${l}_${r}_${c} ============"
      ./test_nsg_index ${base} ${efanna} ${l} ${r} ${c} ${dataset}.nsg
      declare -i lk=$K*$L_SIZE
      echo "Perform kNN searching using NSG index (L${lk}K${K})"
      echo "************ baseline ************"
      ./baseline ${base} ${query} ${dataset}.nsg ${lk} ${K} ${dataset}_nsg_result.ivecs ${gt}
      echo "============== ours =============="
      ./ours ${base} ${query} ${dataset}.nsg ${lk} ${K} ${dataset}_nsg_result.ivecs ${gt}
    done
  done
done

# dataset="crawl"
# base="crawl/crawl_base.fvecs"
# query="crawl/crawl_query.fvecs"
# gt="crawl/crawl_groundtruth.ivecs"
# K=10
# L_SIZE=100
# efanna="crawl_400nn.knng"
# L=( 50 )
# R=( 40 )
# C=( 1000 )

# for l in ${L[@]}; do
#   for r in ${R[@]}; do
#     for c in ${C[@]}; do
#       echo "============ ${l}_${r}_${c} ============"
#       ./test_nsg_index ${base} ${efanna} ${l} ${r} ${c} ${dataset}.nsg
#       declare -i lk=$K*$L_SIZE
#       echo "Perform kNN searching using NSG index (L${lk}K${K})"
#       echo "************ baseline ************"
#       ./baseline ${base} ${query} ${dataset}.nsg ${lk} ${K} ${dataset}_nsg_result.ivecs ${gt}
#       echo "============== ours =============="
#       ./ours ${base} ${query} ${dataset}.nsg ${lk} ${K} ${dataset}_nsg_result.ivecs ${gt}
#     done
#   done
# done
