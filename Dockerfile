FROM quay.io/ascend/cann:9.1.0-beta.3-910b-openeuler24.03-py3.12

ENV LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH"

RUN ln -sf /usr/local/python3.12.13/bin/python3.12-config /usr/local/python3.12.13/bin/python3-config

SHELL [ "/bin/bash", "-c" ]

RUN yum update -y && \
    yum install -y \
        openssl-devel \
        protobuf-devel \
    && yum clean all \
    && rm -rf /var/cache/yum \
    && rm -rf /tmp/*

# Copy source
COPY . /opt/finer

WORKDIR /opt/finer

# Install Rustup
ARG RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
ARG RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y
COPY docker/rust-config.toml /root/.cargo/config.toml

ENV PATH="/root/.cargo/bin:$PATH"

# Install HPCKit
ARG HMPI_TYPE=hns

RUN wget -qO /tmp/HPCKit_26.1.RC1_Linux-aarch64.tar.gz \
    https://mirrors.huaweicloud.com/kunpeng/archive/HPC/HPCKit/HPCKit_26.1.RC1.1/HPCKit_26.1.RC1_Linux-aarch64.tar.gz \
    && tar -xaf /tmp/HPCKit_26.1.RC1_Linux-aarch64.tar.gz -C /tmp \
    && bash /tmp/HPCKit_26.1.RC1_Linux-aarch64/install.sh -y --prefix=/opt --hmpi_type=${HMPI_TYPE} \
    && rm -rf /tmp/*

# Build finer
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/HPCKit/latest/setvars.sh --use-gcc && \
    unset BUILD_DIR && \
    CMAKE_BUILD_TYPE=Release bash script/build-third-party.sh && \
    CMAKE_BUILD_TYPE=Release bash script/build-forch.sh && \
    CMAKE_BUILD_TYPE=Release bash script/build-backend.sh

# Build vllm-rs
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/HPCKit/latest/setvars.sh --use-gcc && \
    cd rust-frontend && cargo build --release --bin vllm-rs

ENV PATH="/opt/finer/build/backend:/opt/finer/rust-frontend/target/release:$PATH"

# Inherit CANN's environment setup;
ENTRYPOINT ["/bin/bash", "-c", "\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/cann/share/info/ascendnpu-ir/bin/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} && \
    source /opt/HPCKit/latest/setvars.sh --use-gcc && \
    exec \"$@\"", "--"]

CMD ["/bin/bash"]
