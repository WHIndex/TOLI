# Toli

Toli is a comprehensive benchmarking suite designed to evaluate the performance of learned and traditional indexes. It focuses on throughput/latency and size under various workloads, allowing you to configure the read/write ratio and test with datasets of different characteristics.

## Prerequisites

To build and use Toli, ensure your system meets the following requirements:

- **GCC** 8.3.0 or later
- **CMake** 3.14.0 or later

### Required Dependencies

- Intel MKL (2018.4.274)
- Intel TBB (2020.3)
- jemalloc

## Build Instructions

1. Create and navigate to the build directory:

   ```bash
   mkdir -p build
   cd build
   ```

2. Configure and build the project:

   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release .. && make
   ```

## Quick Start

To begin benchmarking with Toli, you can calculate throughput using the following command:

```bash
./build/microbench \
--keys_file=./data/dataset \
--keys_file_type={binary,text} \
--read=0.5 --insert=0.5 \
--operations_num=800000000 \
--table_size=-1 \
--init_table_ratio=0.5 \
--thread_num=24 \
--index=index_name \
```

### Explanation of Arguments:

- `--keys_file`: Path to the dataset (can be binary or text format).
- `--keys_file_type`: Specify the dataset type (`binary` or `text`).
- `--read`: Read operation ratio (e.g., `0.5` for 50% reads).
- `--insert`: Insert operation ratio (e.g., `0.5` for 50% inserts).
- `--operations_num`: Total number of operations to run.
- `--table_size`: Set the table size (set to `-1` to infer from the dataset).
- `--init_table_ratio`: Ratio of the dataset to pre-load into the index at the start.
- `--thread_num`: Number of threads to use for benchmarking.
- `--index`: The index to test (specify the index name).

### Additional Options:

- **Latency Measurement**:  
   To collect latency samples, use the `--latency_sample` flag:

   ```bash
   --latency_sample --latency_sample_ratio=0.01
   ```

- **Range Queries**:  
   If you'd like to test range queries (e.g., scanning 100 entries), use:

   ```bash
   --scan_ratio=1 --scan_num=100
   ```

- **Zipfian Distribution**:  
   To perform lookups with a Zipfian distribution:

   ```bash
   --sample_distribution=zipf
   ```

- **Data Shift Experiment**:  
   To preserve the original order in the dataset (no shuffling of keys), enable data-shift mode:

   ```bash
   --data_shift
   ```

- **Dataset Statistics**:  
   To calculate dataset hardness using the PLA-metric with an error bound, use:

   ```bash
   --dataset_statistic --error_bound=32
   ```

- **Memory Consumption Measurement**:  
   If your index supports memory consumption tracking, you can enable it with:

   ```bash
   --memory
   ```

All results will be saved in a CSV file, the location of which is specified via the `--output_path` flag.

