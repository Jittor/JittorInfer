cmake -B build -DGGML_CUDA=OFF -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2" && cmake --build build -j64
# cmake -B build \
#     -DCMAKE_C_COMPILER=/usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha001/aarch64-linux/ccec_compiler/bin/bisheng \
#     -DCMAKE_CXX_COMPILER=/usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha001/aarch64-linux/ccec_compiler/bin/bisheng \
#     -DGGML_CUDA=OFF \
#     -DGGML_CANN=on \
#     -DCMAKE_BUILD_TYPE=Debug \
#     -DLLAMA_MPI=OFF \
#     -DCMAKE_SHARED_LINKER_FLAGS="-lm -lstdc++ -lgcc_s" \
#     -DCMAKE_EXE_LINKER_FLAGS="-lm -lstdc++ -lgcc_s" \
#     -DCMAKE_CXX_FLAGS="-O2 --cce-soc-version=Ascend910B3 --cce-soc-core-type=CubeCore -mcpu=tsv110 --std=c++17" &&\
#     cmake --build build -j64