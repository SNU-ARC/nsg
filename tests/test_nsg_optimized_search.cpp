//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>

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

void load_data_ivecs(char* filename, unsigned int*& data, unsigned& num,
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
  if (argc != 10) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path ground_truth_path hash_bitwidth threshold_percent"
              << std::endl;
    exit(-1);
  }
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
//  std::cout << "ground_truth_num: " << ground_truth_num << ", ground_truth_dim: " << ground_truth_dim << std::endl;
//  for (unsigned int tmp = 0; tmp < ground_truth_num; tmp++) {
//    for (unsigned int tmp1 = 0; tmp1 < ground_truth_dim; tmp1++) {
//      std::cout << *(ground_truth_load + ground_truth_dim * tmp + tmp1 * sizeof(int)) << std::endl;
//    }
//    std::cout << std::endl << std::endl;
//  }
//  assert(query_dim == ground_truth_dim);
#endif
  data_load = efanna2e::data_align(data_load, points_num, dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[3]);
#ifdef THETA_GUIDED_SEARCH
  index.hash_bitwidth = (unsigned)atoi(argv[8]);
  index.threshold_percent = (float)atof(argv[9]);
#else
  index.hash_bitwidth = 0;
#endif
  index.OptimizeGraph(data_load);
#ifdef THETA_GUIDED_SEARCH
  // SJ: For profile, related with #THETA_GUIDED_SEARCH flag
  char* hash_function_name = new char[strlen(argv[3]) + strlen(".hash_function_") + strlen(argv[9]) + 1];
  char* hash_vector_name = new char[strlen(argv[3]) + strlen(".hash_vector") + strlen(argv[9]) + 1];
  strcpy(hash_function_name, argv[3]);
  strcat(hash_function_name, ".hash_function_");
  strcat(hash_function_name, argv[8]);
  strcat(hash_function_name, "b");
  strcpy(hash_vector_name, argv[3]);
  strcat(hash_vector_name, ".hash_vector_");
  strcat(hash_vector_name, argv[8]);
  strcat(hash_vector_name, "b");

  if (index.LoadHashFunction(hash_function_name)) {
    if (!index.LoadHashValue(hash_vector_name))
      index.GenerateHashValue(hash_vector_name);
  }
  else {
    index.GenerateHashFunction(hash_function_name);
    index.GenerateHashValue(hash_vector_name);
  }
  index.theta_queue.reserve(32);
  for (int i = 0; i < 32; i++)
    index.theta_queue[i].distance = -1;
  index.hashed_query = new unsigned int[index.hash_bitwidth >> 5];
#endif
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  auto s = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff;
#pragma omp parallel for num_threads(1)
  for (unsigned i = 0; i < query_num; i++) {
#ifdef THETA_GUIDED_SEARCH
    for (unsigned int a = 0; a < (index.hash_bitwidth >> 5) * query_dim; a += 8) {
      _mm_prefetch(&index.hash_function[a], _MM_HINT_T0);
    }
#endif
   index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();
  diff = e - s;
#ifdef PROFILE
  std::cout << "query_hash time: " << index.profile_time[0].count() << std::endl;
  std::cout << "hash_approx time: " << index.profile_time[1].count() << std::endl;
  std::cout << "dist time: " << index.profile_time[2].count() << std::endl;
#endif

// Print result
  std::cout << "search time: " << diff.count() << "\n";
  std::cout << "QPS: " << query_num / diff.count() << "\n";

  save_result(argv[6], res);

#ifdef GET_MISS_TRAVERSE
  std::cout << std::endl;
  printf("[Total_summary] # of traversed: %u, # of invalid: %u, ratio: %.2f%%\n", index.total_traverse, index.total_traverse_miss, (float)index.total_traverse_miss / index.total_traverse * 100);
#endif
#ifdef EVAL_RECALL
  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    unsigned int topk_local_hit = 0;
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

// Print result for sweep
// #ifdef EVAL_RECALL
//   unsigned int topk_hit = 0;
//   for (unsigned int i = 0; i < query_num; i++) {
//     unsigned int topk_local_hit = 0;
//     for (unsigned int j = 0; j < K; j++) {
//       for (unsigned int k = 0; k < K; k++) {
//         if (res[i][j] == *(ground_truth_load + i * ground_truth_dim + k)) {
//           topk_hit++;
//           break;
//         }
//       }
//     }
//   }
//   std::cout << (float)topk_hit / (query_num * K) * 100 << std::endl;
// #endif
// printf("%u\n%u\n%.2f\n", index.total_traverse, index.total_traverse_miss, (float)index.total_traverse_miss / index.total_traverse * 100);
// std::cout << diff.count() << std::endl;

#ifdef THETA_GUIDED_SEARCH
  delete[] hash_function_name;
  delete[] hash_vector_name;
  delete[] index.hashed_query;
//  index.DeallocateHashVector();
#endif

  return 0;
}
