set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

#add_executable(baseline test_nsg_optimized_search.cpp)
#target_link_libraries(baseline baselinelib -ltcmalloc)

#add_executable(exact test_nsg_optimized_search.cpp)
#target_link_libraries(exact exactlib -ltcmalloc)

#add_executable(ours test_nsg_optimized_search.cpp)
#target_link_libraries(ours ourslib -ltcmalloc)

#target_compile_definitions(baseline PRIVATE GET_MISS_TRAVERSE EVAL_RECALL)
#target_compile_definitions(exact PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH EXACT_SEARCH)
#target_compile_definitions(ours PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH)

