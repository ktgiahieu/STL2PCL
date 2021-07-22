cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(STL2PCD)

add_subdirectory(include)

add_executable (STL2PCD STL2PCD.cpp)
target_link_libraries (STL2PCD HPR meshSampling)
target_include_directories(STL2PCD PUBLIC ${PCL_INCLUDE_DIRS} ${MESHSAMPLING_INCLUDE_DIR} ${HPR_INCLUDE_DIR})
