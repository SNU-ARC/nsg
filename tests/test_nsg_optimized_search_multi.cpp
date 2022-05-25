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
  if (argc != 11) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path ground_truth_path hash_bitwidth threshold_percent num_of_thread"
              << std::endl;
    exit(-1);
  }
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // SJ: Here multi graph starts
  unsigned int num_threads = atoi(argv[10]);
  std::cout << "num_threads: " << num_threads << std::endl;
  omp_set_num_threads(num_threads);
  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K * 16);
  std::vector<double> global_search_time(16, 0.0);

#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned int iter = 0; iter < 16; iter++) {
    float* data_load = NULL;
    unsigned points_num, dim;
    char iter_char[3];
    std::sprintf(iter_char, "%d", iter);
    unsigned int dataname_len = strlen(argv[1]);
    char* sub_dataname = new char[dataname_len + 5];
    strncpy(sub_dataname, argv[1], dataname_len - 6);
    sub_dataname[dataname_len - 6] = '\0';
    strcat(sub_dataname, "_");
    strcat(sub_dataname, iter_char);
    strcat(sub_dataname, &argv[1][dataname_len - 6]);
    std::cout << "Data Path: " << sub_dataname << std::endl;

    load_data(sub_dataname, data_load, points_num, dim);
    data_load = efanna2e::data_align(data_load, points_num, dim);

    assert(dim == query_dim);

    unsigned int indexname_len = strlen(argv[3]);
    char* sub_indexname = new char[indexname_len + 5];
    strncpy(sub_indexname, argv[3], indexname_len - 4);
    sub_indexname[indexname_len - 4] = '\0';
    strcat(sub_indexname, "_");
    strcat(sub_indexname, iter_char);
    strcat(sub_indexname, &argv[3][indexname_len - 4]);
    std::cout << "NSG Path: " << sub_indexname << std::endl;
    std::cout << "Result Path: " << argv[6] << std::endl;

    efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
    index.Load(sub_indexname);
#ifdef THETA_GUIDED_SEARCH
    index.hash_bitwidth = (unsigned)atoi(argv[8]);
    index.threshold_percent = (float)atof(argv[9]);
#else
    index.hash_bitwidth = 0;
#endif
    index.OptimizeGraph(data_load);
#ifdef THETA_GUIDED_SEARCH
    // SJ: For profile, related with #THETA_GUIDED_SEARCH flag
    char* hash_function_name = new char[strlen(sub_indexname) + strlen(".hash_function_") + strlen(argv[8]) + 2];
    char* hash_vector_name = new char[strlen(sub_indexname) + strlen(".hash_vector_") + strlen(argv[8]) + 2];
    strcpy(hash_function_name, sub_indexname);
    strcat(hash_function_name, ".hash_function_");
    strcat(hash_function_name, argv[8]);
    strcat(hash_function_name, "b\0");
    strcpy(hash_vector_name, sub_indexname);
    strcat(hash_vector_name, ".hash_vector_");
    strcat(hash_vector_name, argv[8]);
    strcat(hash_vector_name, "b\0");
    std::cout << hash_function_name << std::endl;
    std::cout << hash_vector_name << std::endl;

    if (index.LoadHashFunction(hash_function_name)) {
      if (!index.LoadHashValue(hash_vector_name))
        index.GenerateHashValue(hash_vector_name);
    }
    else {
      index.GenerateHashFunction(hash_function_name);
      index.GenerateHashValue(hash_vector_name);
    }
#endif
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
#ifdef THETA_GUIDED_SEARCH
      for (unsigned int a = 0; a < (index.hash_bitwidth >> 5) * query_dim; a += 16) {
        _mm_prefetch(&index.hash_function[a], _MM_HINT_T0);
      }
#endif
      index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data() + K * iter);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    global_search_time[iter] += diff.count();

    // SJ: free dynamic alloc arrays
    delete[] data_load;
    delete[] sub_dataname;
    delete[] sub_indexname;
#ifdef THETA_GUIDED_SEARCH
    delete[] hash_function_name;
    delete[] hash_vector_name;
#endif
  }

  std::sort(global_search_time.begin(), global_search_time.end());
  if (num_threads > 1) {
    for (unsigned int iter = 0; iter < 16; iter++) {
      std::cout << iter << "th Search Time: " << global_search_time[iter] << std::endl;
      std::cout << iter << "th QPS: " << query_num / global_search_time[iter] << std::endl;
    }
    std::cout << "Search Time: " << global_search_time[15] << std::endl;
    std::cout << "QPS: " << query_num / global_search_time[15] << std::endl;
  }
  else {
    for (unsigned int iter = 0; iter < 15; iter++) {
      global_search_time[15] += global_search_time[iter];
    }
    std::cout << "Search Time: " << global_search_time[15] << std::endl;
    std::cout << "QPS: " << query_num / global_search_time[15] << std::endl;
  }

#ifdef EVAL_RECALL
  unsigned int* ground_truth_load = NULL;
  unsigned ground_truth_num, ground_truth_dim;
  load_data_ivecs(argv[7], ground_truth_load, ground_truth_num, ground_truth_dim);

  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    unsigned int topk_local_hit = 0;
    for (unsigned int j = 0; j < K * 16; j++) {
      for (unsigned int k = 0; k < K; k++) {
        if (res[i][j] + (6250000 * (j / K)) == *(ground_truth_load + i * ground_truth_dim + k)) {
          topk_hit++;
          break;
        }
      }
    }
  }
  std::cout << (float)topk_hit / (query_num * K) * 100 << "%" << std::endl;
#endif
#ifdef PROFILE
  std::cout << "query_hash time: " << index.profile_time[0].count() << std::endl;
  std::cout << "hash_approx time: " << index.profile_time[1].count() << std::endl;
  std::cout << "dist time: " << index.profile_time[2].count() << std::endl;
#endif

  save_result(argv[6], res);

#ifdef GET_MISS_TRAVERSE
  std::cout << std::endl;
  printf("[Total_summary] # of traversed: %u, # of invalid: %u, ratio: %.2f%%\n", index.total_traverse, index.total_traverse_miss, (float)index.total_traverse_miss / index.total_traverse * 100);
#endif

  return 0;
}
