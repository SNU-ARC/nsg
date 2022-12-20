#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

#include <random>

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

void IndexNSG::SearchWithOptGraph(const float *query, boost::dynamic_bitset<>& flags, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

#ifdef PROFILE
  unsigned int tid = omp_get_thread_num();
  auto visited_list_init_start = std::chrono::high_resolution_clock::now();
#endif
//  [ARC-SJ] Initialize visited list, allocation moved to main module
//  boost::dynamic_bitset<> flags{nd_, 0};
  flags.reset();
#ifdef PROFILE
    auto visited_list_init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> visited_list_init_diff = visited_list_init_end - visited_list_init_start;
    profile_time[tid * 4] += visited_list_init_diff.count() * 1000000;
#endif
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

#ifdef ADA_NNS
#ifdef PROFILE
  auto query_hash_start = std::chrono::high_resolution_clock::now();
#endif
  std::vector<HashNeighbor> selected_pool(100);
  uint64_t hash_size = hash_bitwidth_ >> 5;
  unsigned int* hashed_query = new unsigned int[hash_size];
  QueryHash (query, hashed_query, hash_size);
  unsigned int hash_avx_size = hash_size >> 3;
  __m256i hashed_query_avx[hash_avx_size];
#ifdef __AVX__ 
  for (unsigned int m = 0; m < hash_avx_size; m++) {
    hashed_query_avx[m] = _mm256_loadu_si256((__m256i*)&hashed_query[m << 3]);
  }
#endif
#ifdef PROFILE
  auto query_hash_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> query_hash_diff = query_hash_end - query_hash_start;
  profile_time[tid * 4 + 1] += query_hash_diff.count() * 1000000;
#endif
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

  int k = 0;
  while (k < (int)L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
#ifdef ADA_NNS
#ifdef PROFILE
      auto cand_select_start = std::chrono::high_resolution_clock::now();
#endif
      unsigned selected_pool_size = CandidateSelection(hashed_query, hashed_query_avx, selected_pool, neighbors, MaxM, hash_size);
#ifdef PROFILE
      auto cand_select_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> cand_select_diff = cand_select_end - cand_select_start;
      profile_time[tid * 4 + 2] += cand_select_diff.count() * 1000000;
#endif
#endif
#ifdef PROFILE
      auto dist_start = std::chrono::high_resolution_clock::now();
#endif
#ifdef ADA_NNS
      for (unsigned m = 0; m < selected_pool_size; ++m)
        _mm_prefetch(opt_graph_ + node_size * selected_pool[m].id, _MM_HINT_T0);
      for (unsigned int m = 0; m < selected_pool_size; m++) {
        unsigned int id = selected_pool[m].id;
#else
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
#endif
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
#ifdef GET_DIST_COMP
        total_dist_comp_++;
        if (dist >= retset[L - 1].distance) {
          total_dist_comp_miss_++;
          continue;
        }
#else
        if (dist >= retset[L - 1].distance)
          continue;
#endif
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
#ifdef PROFILE
      auto dist_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dist_diff = dist_end - dist_start;
      profile_time[tid * 4 + 3] += dist_diff.count() * 1000000;
#endif
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

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
#ifdef ADA_NNS
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  uint64_t hash_function_size = dimension_ * hash_bitwidth_ * sizeof(float);
  opt_graph_ = (char *)malloc(node_size * nd_ + hash_len * nd_ + hash_function_size);
#else
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

#ifdef ADA_NNS
void IndexNSG::GenerateHashFunction (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  std::normal_distribution<float> norm_dist (0.0, 1.0);
  std::mt19937 gen(rand());
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
  float hash_function_norm[hash_bitwidth_ - 1];

  std::cout << "GenerateHashFunction" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned int dim = 0; dim < dimension_; dim++) {
    hash_function_[dim] = norm_dist(gen);
  }
  hash_function_norm[0] = dist_fast->norm(hash_function_, dimension_);

  for (unsigned int hash_col = 1; hash_col < hash_bitwidth_; hash_col++) { 
    for (unsigned int dim = 0; dim < dimension_; dim++) {
       hash_function_[hash_col * dimension_ + dim] = norm_dist(gen);
    }

    // Gram-schmidt process
    for (unsigned int compare_col = 0; compare_col < hash_col; compare_col++) {
      float inner_product_between_hash = dist_fast->DistanceInnerProduct::compare(&hash_function_[hash_col * dimension_], &hash_function_[compare_col * dimension_], (unsigned)dimension_);
      for (unsigned int dim = 0; dim < dimension_; dim++) {
        hash_function_[hash_col * dimension_ + dim] -= (inner_product_between_hash / hash_function_norm[compare_col] * hash_function_[compare_col * dimension_ + dim]);
      }
    }
    hash_function_norm[hash_col] = dist_fast->norm(&hash_function_[hash_col * dimension_], dimension_);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashFunction generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hash_function(file_name, std::ios::binary | std::ios::out);
  file_hash_function.write((char*)&hash_bitwidth_, sizeof(unsigned int));
  file_hash_function.write((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
  file_hash_function.close();
}
void IndexNSG::GenerateHashedSet (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  uint64_t hash_len = (hash_bitwidth_ >> 3);

  std::cout << "GenerateHashedSet" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned int i = 0; i < nd_; i++) {
    hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_ + hash_len * i);
    float* vertex = (float *)(opt_graph_ + node_size * i + sizeof(float));
    for (unsigned int num_integer = 0; num_integer < (hash_bitwidth_ >> 5); num_integer++) {
      std::bitset<32> temp_bool;
      for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
        temp_bool.set(bit_count, (dist_fast->DistanceInnerProduct::compare(vertex, &hash_function_[dimension_ * (32 * num_integer + bit_count)], (unsigned)dimension_)) > 0);
      }
      for (unsigned bit_count = 0; bit_count < 32; bit_count++) {
        hashed_set_[num_integer] = (unsigned)(temp_bool.to_ulong());
      }
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashedSet generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hashed_set(file_name, std::ios::binary | std::ios::out);
  hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_);
  for (unsigned int i = 0; i < nd_; i++) {
    for (unsigned int j = 0; j < (hash_len >> 2); j++) { 
      file_hashed_set.write((char*)(hashed_set_ + (hash_len >> 2) * i + j), 4);
    }
  }
  file_hashed_set.close();
}
bool IndexNSG::ReadHashFunction (char* file_name) {
  std::ifstream file_hash_function(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  if (file_hash_function.is_open()) {
    unsigned int hash_bitwidth_temp;
    file_hash_function.read((char*)&hash_bitwidth_temp, sizeof(unsigned int));
    if (hash_bitwidth_ != hash_bitwidth_temp) {
      file_hash_function.close();
      return false;
    }

    std::cout << "ReadHashFunction" << std::endl;
    hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
    file_hash_function.read((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
    file_hash_function.close();
    return true;
  }
  else
    return false;
}
bool IndexNSG::ReadHashedSet (char* file_name) {
  std::ifstream file_hashed_set(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3);
  if (file_hashed_set.is_open()) {
    std::cout << "ReadHashedSet" << std::endl;
    hashed_set_ = (unsigned int*)(opt_graph_ + node_size * nd_);
    for (unsigned int i = 0; i < nd_; i++) {
      for (unsigned int j = 0; j < (hash_len >> 2); j++) {
        file_hashed_set.read((char*)(hashed_set_ + (hash_len >> 2) * i + j), 4);
      }
    }
    file_hashed_set.close();
    
    return true;
  }
  else
    return false;
}
void IndexNSG::QueryHash (const float* query, unsigned* hashed_query, const uint64_t hash_size) {
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (uint64_t num_integer = 0; num_integer < hash_size; num_integer++) {
    std::bitset<32> temp_bool;
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      temp_bool.set(bit_count, (dist_fast->DistanceInnerProduct::compare(query, &hash_function_[dimension_ * (32 * num_integer + bit_count)], dimension_) > 0));
    }
    for (unsigned int bit_count = 0; bit_count < 32; bit_count++) {
      hashed_query[num_integer] = (unsigned)(temp_bool.to_ulong());
    }
  }
}
unsigned IndexNSG::CandidateSelection (const unsigned* hashed_query, const __m256i* hashed_query_avx, std::vector<HashNeighbor>& selected_pool, const unsigned* neighbors, const unsigned MaxM, const uint64_t hash_size) {
  unsigned prefetch_counter = 0;
  for (; prefetch_counter < (MaxM >> 2); ++prefetch_counter) {
    unsigned int id = neighbors[prefetch_counter];
    for (uint64_t n = 0; n < hash_size; n += 8)
      _mm_prefetch(hashed_set_ + hash_size * id + n, _MM_HINT_T0);
  }

  unsigned long long hamming_result[4];
  unsigned int selected_pool_size = 0;
  unsigned int selected_pool_size_limit = (unsigned int)ceil(MaxM * tau_);
  HashNeighbor hamming_distance_max(0, 0);
  std::vector<HashNeighbor>::iterator index;

  for (unsigned m = 0; m < MaxM; ++m) {
    if (prefetch_counter < MaxM) {
      unsigned int id = neighbors[prefetch_counter];
      for (uint64_t n = 0; n < hash_size; n += 8)
        _mm_prefetch(hashed_set_ + hash_size * id + n, _MM_HINT_T0);
      prefetch_counter++;
    }
    unsigned int id = neighbors[m];
    unsigned int hamming_distance = 0;
    unsigned int* hashed_set_address = hashed_set_ + hash_size * id;
#ifdef __AVX__
    for (unsigned int i = 0; i < (hash_size >> 3); i++) {
      __m256i hashed_set_avx;
      __m256i hamming_result_avx;
      hashed_set_avx = _mm256_loadu_si256((__m256i*)(hashed_set_address));
      hamming_result_avx = _mm256_xor_si256(hashed_query_avx[i], hashed_set_avx);
#ifdef __AVX512VPOPCNTDQ__
      hamming_result_avx = _mm256_popcnt_epi64(hamming_result_avx);
      _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
      for (unsigned int j = 0; j < 4; j++)
        hamming_distance += hamming_result[j];
#else
      _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
      for (unsigned int j = 0; j < 4; j++)
        hamming_distance += _mm_popcnt_u64(hamming_result[j]);
#endif
      hashed_set_address += 8;
    }
#else
    for (unsigned int num_integer = 0; num_integer < hash_bitwidth_ / (8 * sizeof(unsigned int)); num_integer++) {
      unsigned int* hashed_set = (unsigned int*)(opt_graph_ + node_size * id + data_len + neighbor_len);
      hamming_result[num_integer] = hashed_query[num_integer] ^ hashed_set[num_integer]; 
      hamming_distance += __builtin_popcount(hamming_result[num_integer]);
    }
#endif
    HashNeighbor cat_hamming_id(id, hamming_distance);
    if ((selected_pool_size_limit < selected_pool_size) && (hamming_distance < hamming_distance_max.distance)) {
      selected_pool[selected_pool_size] = selected_pool[hamming_distance_max.id];
      selected_pool[hamming_distance_max.id] = cat_hamming_id;
      index = std::max_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size_limit);
      hamming_distance_max.id = std::distance(selected_pool.begin(), index);
      hamming_distance_max.distance = selected_pool[hamming_distance_max.id].distance;
      selected_pool_size++;
    }
    else {
      selected_pool[selected_pool_size] = cat_hamming_id;
      selected_pool_size++;
      if (selected_pool_size == selected_pool_size_limit) {
        index = std::max_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size_limit);
        hamming_distance_max.id = std::distance(selected_pool.begin(), index);
        hamming_distance_max.distance = selected_pool[hamming_distance_max.id].distance;
      }
    }
  }
  return selected_pool_size_limit;
}
#endif
}
