# #! /usr/bin/env bash
# git submodule update --init --recursive

mkdir -p src/competitor/rmi/rmi_data


function build_rmi_set() {
    DATA_NAME=$1
    HEADER_PATH=src/competitor/rmi/${DATA_NAME}_0.h
    JSON_PATH=script/rmi_specs/${DATA_NAME}.json

    shift 1
    if [ ! -f $HEADER_PATH ]; then
        echo "Building RMI set for $DATA_NAME"
        src/competitor/rmi/RMI/target/release/rmi --optimize ${JSON_PATH} datasets/datasets_rmi/$DATA_NAME --threads 32
        src/competitor/rmi/RMI/target/release/rmi datasets/datasets_rmi/$DATA_NAME --param-grid ${JSON_PATH} -d src/competitor/rmi/rmi_data/ --threads 32
        mv ${DATA_NAME}_* src/competitor/rmi/
    fi
}


# cd RMI && cargo build --release && cd ..

cd src/competitor/rmi/RMI && cargo build --release && cd ../../../..


build_rmi_set libio_200M_uint64
build_rmi_set fb_200M_uint64
build_rmi_set fb-1_200M_uint64
build_rmi_set genome_200M_uint64
build_rmi_set osm_200M_uint64


bash script/rmi_specs/gen.sh
