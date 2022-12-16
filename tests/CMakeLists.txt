set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_optimized_search_ADA_NNS test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search_ADA_NNS ${PROJECT_NAME}_ADA_NNS -ltcmalloc)
target_compile_definitions(test_nsg_optimized_search_ADA_NNS PRIVATE ADA_NNS)

#add_executable(test_nsg_optimized_search_multi test_nsg_optimized_search_multi.cpp)
#target_link_libraries(test_nsg_optimized_search_multi ${PROJECT_NAME} -ltcmalloc)
