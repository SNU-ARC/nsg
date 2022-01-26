set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

add_library(baselinelib ${CPP_SOURCES})
add_library(exactlib ${CPP_SOURCES})
add_library(exactaidlib ${CPP_SOURCES})
add_library(ourslib ${CPP_SOURCES})

target_compile_definitions(baselinelib PRIVATE GET_MISS_TRAVERSE EVAL_RECALL)
target_compile_definitions(exactlib PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH EXACT_SORT_SEARCH)
target_compile_definitions(exactaidlib PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH EXACT_AID_SEARCH)
target_compile_definitions(ourslib PRIVATE GET_MISS_TRAVERSE EVAL_RECALL THETA_GUIDED_SEARCH)

#install()
