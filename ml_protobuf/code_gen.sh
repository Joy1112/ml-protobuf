#!/bin/bash
dir=$(cd "$(dirname "$0")";pwd);
cd $dir"/protos";
echo $(pwd)

# Dependencies: python==3.7.0, grpcio==1.31.0, grpcio-tools==1.31.0, protobuf==3.20.0
python -m grpc_tools.protoc --python_out=./ndarray --grpc_python_out=./ndarray --proto_path=./protos/ ndarray.proto
