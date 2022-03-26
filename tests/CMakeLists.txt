set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(baseline test_nsg_optimized_search.cpp)
target_link_libraries(baseline baselinelib -ltcmalloc)

add_executable(exact_sort test_nsg_optimized_search.cpp)
target_link_libraries(exact_sort exactlib -ltcmalloc)

add_executable(exact_aid test_nsg_optimized_search.cpp)
target_link_libraries(exact_aid exactaidlib -ltcmalloc)

add_executable(ours test_nsg_optimized_search.cpp)
target_link_libraries(ours ourslib -ltcmalloc)

#target_compile_definitions(baseline PRIVATE GET_MISS_TRAVERSE EVAL_RECALL)
#target_compile_definitions(exact_sort PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH EXACT_SORT_SEARCH)
#target_compile_definitions(exact_aid PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH EXACT_AID_SEARCH)
#target_compile_definitions(ours PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH)

