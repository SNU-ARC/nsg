//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <omp.h>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

// [ARC-SJ]: Read groundtruth 
void load_data_ivecs(char* filename, unsigned int*& data, unsigned& num,
               unsigned& dim) { 
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new unsigned int[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(const char* filename,
                 std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
#ifdef ADA_NNS
  if (argc != 11) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path ground_truth_path hash_bitwidth tau num_threads"
              << std::endl;
    exit(-1);
  }
#else
  if (argc != 9) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path ground_truth_path num_threads"
              << std::endl;
    exit(-1);
  }
#endif
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);
  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

#ifdef EVAL_RECALL
  unsigned int* ground_truth_load = NULL;
  unsigned ground_truth_num, ground_truth_dim;
  load_data_ivecs(argv[7], ground_truth_load, ground_truth_num, ground_truth_dim);
#endif
  data_load = efanna2e::data_align(data_load, points_num, dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[3]);
#ifdef ADA_NNS
  float tau = (float)atof(argv[8]);
  uint64_t hash_bitwidth = (uint64_t)atoi(argv[9]);
  index.SetHashBitwidth(hash_bitwidth);
  index.SetTau(tau);
#endif
  index.OptimizeGraph(data_load);
#ifdef ADA_NNS
  // SJ: For profile, related with #ADA_NNS flag
  char* hash_function_name = new char[strlen(argv[3]) + strlen(".hash_function_") + strlen(argv[9]) + 1];
  char* hashed_set_name = new char[strlen(argv[3]) + strlen(".hashed_set") + strlen(argv[9]) + 1];
  strcpy(hash_function_name, argv[3]);
  strcat(hash_function_name, ".hash_function_");
  strcat(hash_function_name, argv[9]);
  strcat(hash_function_name, "b");
  strcpy(hashed_set_name, argv[3]);
  strcat(hashed_set_name, ".hashed_set_");
  strcat(hashed_set_name, argv[9]);
  strcat(hashed_set_name, "b");

  if (index.ReadHashFunction(hash_function_name)) {
    if (!index.ReadHashedSet(hashed_set_name))
      index.GenerateHashedSet(hashed_set_name);
  }
  else {
    index.GenerateHashFunction(hash_function_name);
    index.GenerateHashedSet(hashed_set_name);
  }
  delete[] hash_function_name;
  delete[] hashed_set_name;
#endif
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  int num_threads = atoi(argv[10]);
  omp_set_num_threads(num_threads);
#ifdef THREAD_LATENCY
  std::vector<double> latency_stats(query_num, 0);
#endif
#ifdef PROFILE
  index.SetTimer(num_threads);
#endif
  // [ARC-SJ]: Minor optimization of greedy search 
  //           Allocate visited list once
  //           For large-scale dataset (e.g., DEEP100M),
  //           repeated allocation is a huge overhead
  boost::dynamic_bitset<> flags{index.Get_nd(), 0};
  auto s = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff;
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < query_num; i++) {
#ifdef THREAD_LATENCY
    auto query_start = std::chrono::high_resolution_clock::now();
#endif
   index.SearchWithOptGraph(query_load + i * dim, flags, K, paras, res[i].data());
#ifdef THREAD_LATENCY
   auto query_end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> query_diff = query_end - query_start;
   latency_stats[i] = query_diff.count() * 1000000;
#endif
  }
  auto e = std::chrono::high_resolution_clock::now();
  diff = e - s;

// Print result
  std::cout << "search time: " << diff.count() << "\n";
  std::cout << "QPS: " << query_num / diff.count() << "\n";

  save_result(argv[6], res);

#ifdef EVAL_RECALL
  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    for (unsigned int j = 0; j < K; j++) {
      for (unsigned int k = 0; k < K; k++) {
        if (res[i][j] == *(ground_truth_load + i * ground_truth_dim + k)) {
          topk_hit++;
          break;
        }
      }
    }
  }
  std::cout << (float)topk_hit / (query_num * K) * 100 << "%" << std::endl;
#endif
#ifdef THREAD_LATENCY
  std::sort(latency_stats.begin(), latency_stats.end());
  double mean_latency = 0;
  for (uint64_t q = 0; q < query_num; q++) {
    mean_latency += latency_stats[q];
  }
  mean_latency /= query_num;
  std::cout << "mean_latency: " << mean_latency << "ms" << std::endl;
  std::cout << "99% latency: " << latency_stats[(unsigned long long)(0.999 * query_num)] << "ms"<< std::endl;
#endif
#ifdef GET_DIST_COMP
  std::cout << "========Distance Compute Report========" << std::endl;
  std::cout << "# of distance compute: " << index.GetTotalDistComp() << std::endl;
  std::cout << "# of missed distance compute: " << index.GetTotalDistCompMiss() << std::endl;
  std::cout << "Ratio: " << (float)index.GetTotalDistCompMiss() / index.GetTotalDistComp()  * 100 << " %" << std::endl;
  std::cout << "Speedup: " << (float)(index.Get_nd()) * query_num / index.GetTotalDistComp() << std::endl;
  std::cout << "=====================================" << std::endl;
#endif
#ifdef PROFILE
  std::cout << "========Thread Latency Report========" << std::endl;
  double* timer = (double*)calloc(4, sizeof(double));
  for (unsigned int tid = 0; tid < num_threads; tid++) {
    timer[0] += index.profile_time[tid * 4]; // visited list init time
    timer[1] += index.profile_time[tid * 4 + 1]; // query hash stage time
    timer[2] += index.profile_time[tid * 4 + 2]; // candidate selection stage time
    timer[3] += index.profile_time[tid * 4 + 3]; // fast L2 distance compute time
  }
#ifdef ADA_NNS
  std::cout << "visited_init time: " << timer[0] / query_num << "ms" << std::endl;
  std::cout << "query_hash time: " << timer[1] / query_num << "ms" << std::endl;
  std::cout << "cand_select time: " << timer[2] / query_num << "ms" << std::endl;
  std::cout << "dist time: " << timer[3] / query_num << "ms" << std::endl;
#else
  std::cout << "visited_init time: " << timer[0] / query_num << "ms" << std::endl;
  std::cout << "dist time: " << timer[3] / query_num << "ms" << std::endl;
#endif
  std::cout << "=====================================" << std::endl;
#endif


  return 0;
}
