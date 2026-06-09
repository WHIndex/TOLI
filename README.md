# Tali

Tali is a C++17 benchmark harness for learned index structures. The project
centers on a single executable, `microbench`, which runs configurable online
workloads and writes throughput, latency, memory, workload, and index-parameter
metadata to CSV.

## Repository Layout

- `src/benchmark`: workload generation, flag parsing, timing, and CSV reporting.
- `src/competitor`: adapters for the index implementations used by `microbench`.
- `datasets`: download scripts and a synthetic key generator.
- `cmake`: local find modules for external dependencies.

## Supported Indexes

Pass one or more index names with `--index`, separated by commas:

```text
alex, alexol, art, artolc, btree, btreeolc,
dili, dpgm, dytis, finedex, lipp, lippol,
masstree, pgm, sali, xindex
```


## Requirements

- GCC 11.4.0 or newer
- CMake 3.14 or newer
- OpenMP
- Intel MKL
- Intel TBB
- jemalloc
- Boost components: `system`, `thread`, and `chrono`

## Build

Initialize submodules before the first build:

```bash
git submodule update --init --recursive
```

Then configure and compile:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful build-time knobs are exposed as CMake cache variables. For example,
`K_MAX_DENSITY`, `ROOT_ERROR_BOUND`, `LIPP_THRESHOLD1`, `SALI_THRESHOLD1`,
`PGM_EPSILON`, and `BTREE_LEAF_SLOTS` are forwarded into the benchmark binary
as compile-time constants.

## Run

All non-boolean options use `--key=value`. Boolean options are enabled by
including the flag name, such as `--latency_sample` or `--memory`.

```bash
./build/microbench \
  --keys_file=./datasets/covid \
  --keys_file_type=binary \
  --read=0.8 \
  --insert=0.2 \
  --update=0 \
  --delete=0 \
  --scan=0 \
  --operations_num=100000000 \
  --table_size=-1 \
  --init_table_ratio=0.5 \
  --thread_num=24 \
  --index=alex,xindex,sali \
  --output_path=./out.csv
```

Unless a test suite mode is selected, the operation ratios must sum to `1.0`.
`--thread_num` and `--index` accept comma-separated lists, so a single command
can sweep multiple thread counts or index implementations.

## Output

Each run appends one CSV row. The report includes the workload mix, input paths,
index name, throughput, initial table size, memory consumption, thread count,
latency percentiles when sampling is enabled, random seed, scan length, dataset
fitness metrics, partition settings, operation count, and selected per-index
tuning parameters.

## Datasets

Download helper scripts live in `datasets`:

```bash
cd datasets
bash download.sh
```

The synthetic generator can be built independently:

```bash
g++ --std=c++17 generator.cpp -o generator
./generator {le} {ge} {lv} {gv} {num} {path}
```

See `datasets/README.md` for dataset provenance and generator details.
