#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

#include <random>
// SJ: For profile
static bool neighbor_list_flag;
static bool all_vector_flag;
static bool count_zero_element_flag;
static bool count_negative_element_flag;
static bool get_min_max_element_flag;
static unsigned int nth_query = 0;
static unsigned int ntraverse = 0;

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  // std::cout<<cc<<std::endl;
}
void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
}

void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  width = range;
  unsigned start = 0;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_);
      /*
    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
      }
      */
    }
  }

#pragma omp for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_);
  }
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  tree_grow(parameters);

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
  unsigned int tid = omp_get_thread_num();

  std::vector<HashNeighbor> theta_queue(512);
  unsigned int* hashed_query = new unsigned int[hash_bitwidth >> 5];

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

#ifdef GET_NEIGHBOR_LIST
  GetNeighborList();
#endif
#ifdef GET_ALL_VECTOR
  GetAllVectors();
#endif
#ifdef COUNT_ZERO_ELEMENT
  CountZeroElements();
#endif
#ifdef COUNT_NEGATIVE_ELEMENT
  CountNegativeElements();
#endif
#ifdef GET_MIN_MAX_ELEMENT
  GetMinMaxElement();
#endif

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
#ifdef PROFILE
  auto query_hash_start = std::chrono::high_resolution_clock::now();
#endif
#ifdef THETA_GUIDED_SEARCH
  unsigned int hash_size = hash_bitwidth >> 5;
  for (unsigned int num_integer = 0; num_integer < hash_size; num_integer++) {
    std::bitset<32> temp_bool;
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      temp_bool.set(bit_count, (dist_fast->DistanceInnerProduct::compare(query, &hash_function[dimension_ * (32 * num_integer + bit_count)], dimension_) > 0));
    }
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      hashed_query[num_integer] = (unsigned)(temp_bool.to_ulong());
    }
  }
#endif

#ifdef PROFILE
  auto query_hash_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> query_hash_diff = query_hash_end - query_hash_start;
  profile_time[tid * num_timer] += query_hash_diff.count() * 1000000;
#endif

  int k = 0;
#ifdef GET_MISS_TRAVERSE
  unsigned int query_traverse = 0;
  unsigned int query_traverse_miss = 0;
#endif
#ifdef THETA_GUIDED_SEARCH
  __m256i hashed_query_avx[hash_size >> 3];
  for (unsigned int m = 0; m < (hash_size >> 3); m++) {
    hashed_query_avx[m] = _mm256_loadu_si256((__m256i*)&hashed_query[m << 3]);
  }
#endif
  while (k < (int)L) {
    int nk = L;
#ifdef GET_MISS_TRAVERSE
    unsigned int local_traverse = 0;
    unsigned int local_traverse_miss = 0;
    std::vector<unsigned> inserted_ids;
#endif
#ifdef THETA_GUIDED_SEARCH
    unsigned int local_far_neighbors = 0;
#endif
    int r;

    if (retset[k].flag) {

#ifdef PROFILE
      auto hash_approx_start = std::chrono::high_resolution_clock::now();
#endif
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
#ifdef THETA_GUIDED_SEARCH
      unsigned long long hamming_result[4];
      unsigned int theta_queue_size = 0;
      unsigned int theta_queue_size_limit = (unsigned int)ceil(MaxM * threshold_percent);
      HashNeighbor hamming_distance_max(0, 0);
      std::vector<HashNeighbor>::iterator index;
     
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned int id = neighbors[m];
//        _mm_prefetch((unsigned*)(hash_value + hash_size * id), _MM_HINT_T0);
        for (unsigned k = 0; k < hash_size; k++)
          _mm_prefetch(hash_value + hash_size * id + k, _MM_HINT_T0);
      }
#endif
#ifdef THETA_GUIDED_SEARCH
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned int id = neighbors[m];
        unsigned int hamming_distance = 0;
#ifdef __AVX__
        unsigned int* hash_value_address = (unsigned int*)(opt_graph_ + node_size * nd_ + hash_len * id);
        for (unsigned int i = 0; i < (hash_size >> 3); i++) {
          __m256i hash_value_avx, hamming_result_avx;
          hash_value_avx = _mm256_loadu_si256((__m256i*)(hash_value_address));
          hamming_result_avx = _mm256_xor_si256(hashed_query_avx[i], hash_value_avx);
#ifdef __AVX512VPOPCNTDQ__
          hamming_result_avx = _mm256_popcnt_epi64(hamming_result_avx);
          _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
          for (unsigned int j = 0; j < 4; j++)
            hamming_distance += hamming_result[j];
#else
          _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
          for (unsigned int j = 0; j < 4; j++)
            hamming_distance += _popcnt64(hamming_result[j]);
          hash_value_address += 8;
#endif
        }
#else
        for (unsigned int num_integer = 0; num_integer < hash_bitwidth / (8 * sizeof(unsigned int)); num_integer++) {
          unsigned int* hash_value_address = (unsigned int*)(opt_graph_ + node_size * id + data_len + neighbor_len);
          hamming_result[num_integer] = hashed_query[num_integer] ^ hash_value_address[num_integer]; 
          hamming_distance += __builtin_popcount(hamming_result[num_integer]);
        }
#endif
        HashNeighbor cat_hamming_id(id, hamming_distance);
//        InsertIntoPool (theta_queue.data(), theta_queue_size_limit, cat_hamming_id);
        if ((theta_queue_size < theta_queue_size_limit) || (hamming_distance == hamming_distance_max.distance)) {
          theta_queue[theta_queue_size] = cat_hamming_id;
          theta_queue_size++;
          index = std::max_element(theta_queue.begin(), theta_queue.begin() + theta_queue_size_limit);
          hamming_distance_max.id = std::distance(theta_queue.begin(), index);
          hamming_distance_max.distance = theta_queue[hamming_distance_max.id].distance;
        }
        else if (hamming_distance < hamming_distance_max.distance) {
          theta_queue[hamming_distance_max.id] = cat_hamming_id;
          theta_queue_size = theta_queue_size_limit;
          index = std::max_element(theta_queue.begin(), theta_queue.begin() + theta_queue_size);
          hamming_distance_max.id = std::distance(theta_queue.begin(), index);
          hamming_distance_max.distance = theta_queue[hamming_distance_max.id].distance;
        }
      }
#endif
      for (unsigned m = 0; m < MaxM; ++m) 
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);


#ifdef PROFILE
      auto hash_approx_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hash_approx_diff = hash_approx_end - hash_approx_start;
      profile_time[tid * num_timer + 1] += hash_approx_diff.count() * 1000000;
      auto dist_start = std::chrono::high_resolution_clock::now();
#endif
#ifdef THETA_GUIDED_SEARCH
//      std::cout << "theta_queue_size: " << theta_queue_size << ", theta_queue_size_limit: " << theta_queue_size_limit << std::endl;
      for (unsigned int m = 0; m < theta_queue_size; m++) {
        unsigned int id = theta_queue[m].id;
        theta_queue[m].distance = -1;
#else
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
#endif
        if (flags[id]) continue;
        flags[id] = 1;
        ntraverse++;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) {
#ifdef GET_MISS_TRAVERSE
          local_traverse++;
          query_traverse++;
          local_traverse_miss++;
          query_traverse_miss++;
#endif
          continue;
        }
#ifdef GET_MISS_TRAVERSE
        local_traverse++;
        query_traverse++;
#endif
        Neighbor nn(id, dist, true);
        r = InsertIntoPool(retset.data(), L, nn);
//        int r = InsertIntoPool(retset.data(), L, nn);

#ifdef GET_NORM_VS_RANK
        printf("norm: %f, dist: %f, diff: %f, rank: %d\n", norm, dist, (norm-dist)/2, r);
#endif
#ifdef GET_MISS_TRAVERSE
        inserted_ids.push_back(id);
#endif
        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
#ifdef PROFILE
      auto dist_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dist_diff = dist_end - dist_start;
      profile_time[tid * num_timer + 2] += dist_diff.count() * 1000000;
#endif
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
#ifdef GET_NORM_VS_RANK
  printf("\n\n");
#endif
  }
#ifdef GET_TOPK_SNAPSHOT
    printf("%u query result\n", nth_query);
#endif
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
#ifdef GET_TOPK_SNAPSHOT
    printf("id = %u, distance = %f, theta = %f\n", retset[i].id, retset[i].distance, retset[i].theta);
#endif
  }
#ifdef GET_MISS_TRAVERSE
  total_traverse += query_traverse;
  total_traverse_miss += query_traverse_miss;
//  printf("[Query_summary] # of traversed: %u, # of invalid: %u, ratio: %.2f%%\n", query_traverse, query_traverse_miss, (float)query_traverse_miss / query_traverse * 100);
#endif
  nth_query++;
}

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
#ifdef THETA_GUIDED_SEARCH
  hash_len = (hash_bitwidth >> 3); // SJ: Append hash_values
  node_size = data_len + neighbor_len;
//  node_size = data_len + neighbor_len + hash_len;
  hash_function_size = dimension_ * hash_bitwidth * sizeof(float);
  opt_graph_ = (char *)malloc(node_size * nd_ + hash_len * (nd_ + 1) + hash_function_size);
#else
  node_size = data_len + neighbor_len;
  opt_graph_ = (char *)malloc(node_size * nd_);
#endif
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}
void IndexNSG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    // std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}
// SJ: For profiling
void IndexNSG::GetNeighborList() {
  if (!neighbor_list_flag) {
    printf("# vertices: %lu\n", nd_);
    for (unsigned i = 0; i < nd_; i++) {
      unsigned* neighbors_list = (unsigned *)(opt_graph_ + node_size * i + data_len);
      unsigned num_neighbors = *neighbors_list;
      neighbors_list++;
      for (unsigned j = 0; j < num_neighbors; j++) {
        char* neighbor_addr = opt_graph_ + node_size * neighbors_list[j];
        printf("neighbors_list[%u][%u] id: %u, addr: 0x%lx\n", i, j, neighbors_list[j], (uint64_t)neighbor_addr); 
      }
    }
    neighbor_list_flag = true;
  }
}
void IndexNSG::GetAllVectors() {
  if (!all_vector_flag) {
    for (unsigned i = 0; i < nd_; i++) {
      float* vertex = (float*)(opt_graph_ + node_size * i);
//      float norm = *vertex;
      vertex++;
      for (unsigned j = 0; j < dimension_; j++) {
        printf("%f ", vertex[j]); 
      }
      printf("\n");
    }
    all_vector_flag = true;
  }
}
void IndexNSG::CountZeroElements() {
  if (!count_zero_element_flag) {
    unsigned zero_count = 0;
    printf("# elements: %lu\n", nd_ * dimension_);
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned j = 0; j < dimension_; j++) {
        unsigned is_zero = (*(float*)(opt_graph_ + node_size * i + 4 * (j + 1)) == 0.0);
        zero_count += is_zero; 
      }
    }
    printf("zero_count = %u\n", zero_count);
    printf("zero_rate = %.2lf%%\n\n", (double)zero_count / (nd_ * dimension_));
    count_zero_element_flag = true;
  }
}
void IndexNSG::CountNegativeElements() {
  if (!count_negative_element_flag) {
    unsigned negative_count = 0;
    printf("# elements: %lu\n", nd_ * dimension_);
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned j = 0; j < dimension_; j++) {
        unsigned is_negative = (*(float*)(opt_graph_ + node_size * i + 4 * (j + 1)) < 0.0);
          negative_count += is_negative; 
      }
    }
    printf("negative_count = %u\n", negative_count);
    printf("negative_rate = %.2lf%%\n\n", (double)negative_count / (nd_ * dimension_));
    count_negative_element_flag = true;
  }
}
void IndexNSG::GetMinMaxElement() {
  if (!get_min_max_element_flag) {
    float* max_elements = (float*)calloc(dimension_, sizeof(float));
    float* min_elements = (float*)calloc(dimension_, sizeof(float));
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned j = 0; j < dimension_; j++) {
        float* temp_data = (float*)(opt_graph_ + node_size * i + 4);
        max_elements[j] = max_elements[j] < temp_data[j] ? temp_data[j] : max_elements[j];
        min_elements[j] = min_elements[j] > temp_data[j] ? temp_data[j] : min_elements[j];
      }
    }
    std::cout << "max_elements:";
    for (unsigned i = 0; i < dimension_; i++)
      std::cout << " " << max_elements[i];
    std::cout<<std::endl;
    std::cout << "min_elements:";
    for (unsigned i = 0; i < dimension_; i++)
      std::cout << " " << min_elements[i];
    std::cout<<std::endl;
    get_min_max_element_flag = true;
    free(max_elements);
    free(min_elements);
  }
}

// SJ: For SRP
void IndexNSG::GenerateHashFunction (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  std::normal_distribution<float> norm_dist (0.0, 1.0);
  std::mt19937 gen(rand());
  hash_function = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
//  hash_function = (float*)(opt_graph_ + node_size * nd_);
//  hash_function = new float[dimension_ * hash_bitwidth];
  float hash_function_norm[hash_bitwidth - 1];

  std::cout << "GenerateHashFunction" << std::endl;
  for (unsigned int dim = 0; dim < dimension_; dim++) { // Random generated vector
    hash_function[dim] = norm_dist(gen);
  }
  hash_function_norm[0] = dist_fast->norm(hash_function, dimension_);

  for (unsigned int hash_col = 1; hash_col < hash_bitwidth; hash_col++) { // Iterate to generate vectors orthogonal to 0th column
    for (unsigned int dim = 0; dim < dimension_; dim++) { // Random generated vector
       hash_function[hash_col * dimension_ + dim] = norm_dist(gen);
    }
    hash_function_norm[hash_col] = dist_fast->norm(&hash_function[hash_col * dimension_], dimension_);

    // Gram-schmidt process
    for (unsigned int compare_col = 0; compare_col < hash_col; compare_col++) {
      float inner_product_between_hash = dist_fast->DistanceInnerProduct::compare(&hash_function[hash_col * dimension_], &hash_function[compare_col * dimension_], (unsigned)dimension_);
      for (unsigned int dim = 0; dim < dimension_; dim++) {
        hash_function[hash_col * dimension_ + dim] -= (inner_product_between_hash / hash_function_norm[compare_col] * hash_function[compare_col * dimension_ + dim]);
      }
    }
  }

  std::ofstream file_hash_function(file_name, std::ios::binary | std::ios::out);
  file_hash_function.write((char*)&hash_bitwidth, sizeof(unsigned int));
  file_hash_function.write((char*)hash_function, dimension_ * hash_bitwidth * sizeof(float));
  file_hash_function.close();
}
void IndexNSG::GenerateHashValue (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;

  std::cout << "GenerateHashValue" << std::endl;
  for (unsigned int i = 0; i < nd_; i++) {
    unsigned int* neighbors = (unsigned int*)(opt_graph_ + node_size * i + data_len);
    unsigned int MaxM = *neighbors;
    neighbors++;
    unsigned int* hash_value = (unsigned int*)(opt_graph_ + node_size * i + data_len + neighbor_len);

    float* vertex = (float *)(opt_graph_ + node_size * i + sizeof(float));
    for (unsigned int i = 0; i < hash_bitwidth / (8 * sizeof(unsigned int)); i++) {
      unsigned int hash_value_temp = 0;
      for (unsigned int bit_count = 0; bit_count < (8 * sizeof(unsigned int)); bit_count++) {
        hash_value_temp = hash_value_temp >> 1;
        hash_value_temp = hash_value_temp | (dist_fast->DistanceInnerProduct::compare(vertex, &hash_function[dimension_ * ((8 * sizeof(unsigned int)) * i + bit_count)], dimension_) > 0 ? 0x80000000 : 0);
      }
      hash_value[i] = hash_value_temp;
    }
  }

  std::ofstream file_hash_value(file_name, std::ios::binary | std::ios::out);
  hash_value = (unsigned int*)(opt_graph_ + node_size * nd_); 
  for (unsigned int i = 0; i < nd_; i++) {
//    unsigned int* hash_value = (unsigned int*)(opt_graph_ + node_size * i + data_len + neighbor_len);
    for (unsigned int j = 0; j < (hash_bitwidth >> 5); j++) { 
      file_hash_value.write((char*)(hash_value + (hash_len >> 2) * i + j), 4);
    }
  }
  file_hash_value.close();
}
void IndexNSG::DeallocateHashVector () {
  delete[] hash_function;
}
bool IndexNSG::LoadHashFunction (char* file_name) {
  std::ifstream file_hash_function(file_name, std::ios::binary);
  if (file_hash_function.is_open()) {
    unsigned int hash_bitwidth_temp;
    file_hash_function.read((char*)&hash_bitwidth_temp, sizeof(unsigned int));
    if (hash_bitwidth != hash_bitwidth_temp) {
      file_hash_function.close();
      return false;
    }

    hash_function = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
//    hash_function = (float*)(opt_graph_ + node_size * nd_);
    float* hash_function_temp = new float[dimension_ * hash_bitwidth];
    hash_function_temp = (float*)memalign(32, dimension_ * hash_bitwidth * sizeof(float));
    file_hash_function.read((char*)hash_function_temp, dimension_ * hash_bitwidth * sizeof(float));
    file_hash_function.close();
    memcpy(hash_function, hash_function_temp, dimension_ * hash_bitwidth * sizeof(float));
    delete[] hash_function_temp;
    return true;
  }
  else {
    return false;
  }
}
bool IndexNSG::LoadHashValue (char* file_name) {
  std::ifstream file_hash_value(file_name, std::ios::binary);
  if (file_hash_value.is_open()) {
    hash_value = (unsigned int*)(opt_graph_ + node_size * nd_);
    unsigned int* hash_value_temp = new unsigned int[nd_ * (hash_bitwidth >> 5)];
    hash_value_temp = (unsigned int*)memalign(32, nd_ * (hash_bitwidth >> 3));
    for (unsigned int i = 0; i < nd_; i++) {
      for (unsigned int j = 0; j < (hash_bitwidth >> 5); j++) {
        file_hash_value.read((char*)(hash_value_temp + (hash_len >> 2) * i + j), 4);
      }
    }
    file_hash_value.close();
    memcpy(hash_value, hash_value_temp, nd_ * (hash_bitwidth >> 3));
    delete[] hash_value_temp;
    
    return true;
  }
  else {
    return false;
  }
}
}
