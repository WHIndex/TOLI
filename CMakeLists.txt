cmake_minimum_required(VERSION 3.14)

# 设置编译器
# set(CMAKE_C_COMPILER "/usr/bin/gcc")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(Tali)

# 确保GCC版本不低于11.4.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.4.0)
        message(FATAL_ERROR "GCC version must be at least 11.4.0")
    endif()
endif()

# 添加自定义模块路径
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# 使用FetchContent和ExternalProject管理外部依赖
include(FetchContent)
include(ExternalProject)

# 查找并配置依赖
find_package(OpenMP REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)
find_package(MKL REQUIRED)
# find_package(Boost REQUIRED)
# 指定需要的 Boost 组件
find_package(Boost REQUIRED COMPONENTS system thread chrono)
# 线程库（给 -pthread）
find_package(Threads REQUIRED)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件目录
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})
include_directories(${JEMALLOC_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

add_compile_options(${OpenMP_CXX_FLAGS})

if(PROFILING)
    message("Compiling for profiling")
    add_definitions(-DPROFILING)
endif()

if(ROOT_PROFILING)
    message("Compiling for root profiling")
    add_definitions(-DROOT_PROFILING)
endif()

set(K_MAX_DENSITY 0.8 CACHE STRING "ALEX max density")
set(K_INIT_DENSITY 0.7 CACHE STRING "ALEX init density")
set(K_MIN_DENSITY 0.6 CACHE STRING "ALEX min density")

add_definitions(-DK_MAX_DENSITY=${K_MAX_DENSITY})
add_definitions(-DK_INIT_DENSITY=${K_INIT_DENSITY})
add_definitions(-DK_MIN_DENSITY=${K_MIN_DENSITY})

set(ROOT_ERROR_BOUND 64 CACHE STRING "XIndex root error bound")
set(ROOT_MEMORY_CONSTRAINT 1048576 CACHE STRING "XIndex root memory constraint")
set(GROUP_ERROR_BOUND 64 CACHE STRING "XIndex group error bound")
set(GROUP_ERROR_TOLERANCE 4 CACHE STRING "XIndex group error tolerance")
set(BUFFER_SIZE_BOUND 256 CACHE STRING "XIndex buffer size bound")
set(BUFFER_SIZE_TOLERANCE 3 CACHE STRING "XIndex buffer size tolerance")
set(BUFFER_COMPACT_THRESHOLD 8 CACHE STRING "XIndex buffer compact threshold")
set(SEQ_INSERT_RESERVE_FACTOR 2 CACHE STRING "XIndex seq insert reserve factor")

add_definitions(-DROOT_ERROR_BOUND=${ROOT_ERROR_BOUND})
add_definitions(-DROOT_MEMORY_CONSTRAINT=${ROOT_MEMORY_CONSTRAINT})
add_definitions(-DGROUP_ERROR_BOUND=${GROUP_ERROR_BOUND})
add_definitions(-DGROUP_ERROR_TOLERANCE=${GROUP_ERROR_TOLERANCE})
add_definitions(-DBUFFER_SIZE_BOUND=${BUFFER_SIZE_BOUND})
add_definitions(-DBUFFER_SIZE_TOLERANCE=${BUFFER_SIZE_TOLERANCE})
add_definitions(-DBUFFER_COMPACT_THRESHOLD=${BUFFER_COMPACT_THRESHOLD})
add_definitions(-DSEQ_INSERT_RESERVE_FACTOR=${SEQ_INSERT_RESERVE_FACTOR})

set(LIPP_THRESHOLD1 1000000 CACHE STRING "LIPP threshold1 (default 1M)")
set(LIPP_GAP1        1        CACHE STRING "LIPP gap1 (default 1)")
set(LIPP_THRESHOLD2  100000   CACHE STRING "LIPP threshold2 (default 100K)")
set(LIPP_GAP2        2        CACHE STRING "LIPP gap2 (default 2)")
set(LIPP_DEFAULT_GAP 5        CACHE STRING "LIPP default gap (default 5)")

add_definitions(-DLIPP_THRESHOLD1=${LIPP_THRESHOLD1})
add_definitions(-DLIPP_GAP1=${LIPP_GAP1})
add_definitions(-DLIPP_THRESHOLD2=${LIPP_THRESHOLD2})
add_definitions(-DLIPP_GAP2=${LIPP_GAP2})
add_definitions(-DLIPP_DEFAULT_GAP=${LIPP_DEFAULT_GAP})

set(SALI_THRESHOLD1 1000000 CACHE STRING "SALI threshold1 (default 1M)")
set(SALI_GAP1        1        CACHE STRING "SALI gap1 (default 1)")
set(SALI_THRESHOLD2  100000   CACHE STRING "SALI threshold2 (default 100K)")
set(SALI_GAP2        2        CACHE STRING "SALI gap2 (default 2)")
set(SALI_DEFAULT_GAP 5        CACHE STRING "SALI default gap (default 5)")

add_definitions(-DSALI_THRESHOLD1=${SALI_THRESHOLD1})
add_definitions(-DSALI_GAP1=${SALI_GAP1})
add_definitions(-DSALI_THRESHOLD2=${SALI_THRESHOLD2})
add_definitions(-DSALI_GAP2=${SALI_GAP2})
add_definitions(-DSALI_DEFAULT_GAP=${SALI_DEFAULT_GAP})

set(PGM_EPSILON 64 CACHE STRING "PGM main Epsilon (error bound)")
set(PGM_EPSILON_RECURSIVE 4 CACHE STRING "PGM recursive Epsilon")

add_definitions(-DPGM_EPSILON=${PGM_EPSILON})
add_definitions(-DPGM_EPSILON_RECURSIVE=${PGM_EPSILON_RECURSIVE})

set(FINEDEX_MAX_ERR 64 CACHE STRING "Finedex maxErr (error bound)")

add_definitions(-DFINEDEX_MAX_ERR=${FINEDEX_MAX_ERR})

set(ART_MAX_PREFIX_LEN 10 CACHE STRING "ART max stored prefix length")
add_definitions(-DART_MAX_PREFIX_LEN=${ART_MAX_PREFIX_LEN})
set(NODE4_MAX 4 CACHE STRING "ART Node4 max size")
add_definitions(-DNODE4_MAX=${NODE4_MAX})
set(NODE16_MAX 32 CACHE STRING "ART Node16 max size")
add_definitions(-DNODE16_MAX=${NODE16_MAX})
set(NODE48_MAX 64 CACHE STRING "ART Node48 max size")
add_definitions(-DNODE48_MAX=${NODE48_MAX})

set(BTREE_LEAF_SLOTS 8 CACHE STRING "B+tree leaf slots")
set(BTREE_INNER_SLOTS 8 CACHE STRING "B+tree inner slots")
set(BTREE_BINSEARCH_THRESHOLD 256 CACHE STRING "B+tree binsearch threshold")
set(BTREE_MIN_FACTOR 2 CACHE STRING "B+tree min factor for minleafslots/mininnerslots (default 2)")

add_definitions(-DBTREE_LEAF_SLOTS=${BTREE_LEAF_SLOTS})
add_definitions(-DBTREE_INNER_SLOTS=${BTREE_INNER_SLOTS})
add_definitions(-DBTREE_BINSEARCH_THRESHOLD=${BTREE_BINSEARCH_THRESHOLD})
add_definitions(-DBTREE_MIN_FACTOR=${BTREE_MIN_FACTOR})

set(BTREEOLC_PAGESIZE 512 CACHE STRING "BTreeOLC node page size in bytes")
add_definitions(-DBTREEOLC_PAGESIZE=${BTREEOLC_PAGESIZE})

set(LEAF_MAX_CAPACIY 8192 CACHE STRING "DILI leaf max capacity")
set(DILI_FAN_THRESHOLD 8192 CACHE STRING "DILI fan threshold")
set(DILI_RHO 0.1 CACHE STRING "DILI decay factor rho")
set(DILI_BU_MIN_FAN 16 CACHE STRING "DILI bottom-up min fan")
add_definitions(-DLEAF_MAX_CAPACIY=${LEAF_MAX_CAPACIY})
add_definitions(-DDILI_FAN_THRESHOLD=${DILI_FAN_THRESHOLD})
add_definitions(-DDILI_RHO=${DILI_RHO})
add_definitions(-DDILI_BU_MIN_FAN=${DILI_BU_MIN_FAN})

include_directories(src/competitor/rmi)
file(GLOB RMI_SOURCES "src/competitor/rmi/*.cpp")
add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp ${RMI_SOURCES})

# 定义microbench可执行文件
# add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp)

# 针对不同构建类型设置编译选项
target_compile_options(microbench PRIVATE
    $<$<CONFIG:DEBUG>:-g -Og -march=native -faligned-new>
    $<$<CONFIG:RELEASE>:-O3 -march=native -faligned-new>
    $<$<CONFIG:OPTI_ENHANCE>:-O3 -march=native -faligned-new -flto -funroll-all-loops>
)

add_subdirectory(src/competitor/masstree)

target_link_libraries(microbench PRIVATE
    MASSTREE
)

# ---- 关键：把需要的 Boost/Threads 链接上 ----
# 推荐用导入目标（CMake 3.14 可用）
target_link_libraries(microbench PRIVATE
    Boost::system
    Boost::thread
    Boost::chrono
    Threads::Threads
)

# 如果你的环境里没有 Boost::xxx 这些导入目标（极少数旧打包方式会这样），
# 也可以追加下面这一行作为兜底：
# target_link_libraries(microbench PRIVATE ${Boost_LIBRARIES})

# 链接依赖库
target_link_libraries(microbench PUBLIC 
    ${OpenMP_CXX_LIBRARIES}
    ${JEMALLOC_LIBRARIES}
    ${TBB_LIBRARIES}
    ${MKL_LIBRARIES}
)
