#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.6)

#Name your project here
project(DiffusionModel)

set(DIFFUSION_VERSION 0.1)

SET(BUILD_RELEASE ON)
SET(BUILD_DEBUG OFF)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD 11)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        list(APPEND CMAKE_PREFIX_PATH "/services/tools/armadillo/6.500.5/" "${CMAKE_SOURCE_DIR}/../../nlopt/")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

FIND_PACKAGE(Nlopt REQUIRED)
FIND_PACKAGE(Armadillo REQUIRED)

IF(BUILD_RELEASE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -fmessage-length=0")
ENDIF(BUILD_RELEASE)

IF(BUILD_DEBUG)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -fmessage-length=0")
ENDIF(BUILD_DEBUG)

set(PROGS main ParameterSearch)

add_library(diffusion STATIC diffusion.cpp)

FOREACH(p ${PROGS})
	add_executable(${p} ${p}.cpp)
	target_link_libraries(${p} diffusion ${ARMADILLO_LIBRARIES} ${NLOPT_LIBRARY})
ENDFOREACH(p)
