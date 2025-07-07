cmake_minimum_required(VERSION 3.12.0)
project(FishKolm LANGUAGES CXX C)

include(../common/cmake-common.cmake)



add_executable(FishKolm src/main.cpp src/FishKolm.cpp)
deal_ii_setup_target(FishKolm)
