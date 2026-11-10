#pragma once

/// @file transport.hpp
/// @brief Frontend ZMQ link plus the TP request/token fan-out.
///
/// Only rank0 owns ZMQ. Every rank owns a Transport and calls the same
/// methods in the same order, so the scheduler never branches on rank:
/// rank0 receives, then broadcasts; followers receive the broadcast; outbound
/// messages are dropped on followers.

#include <cerrno>
#include <cstring>
#include <mpi.h>

#include <spdlog/spdlog.h>
#include <zmq.h>

#include "config.hpp"
#include "output-sink.hpp"
#include "protocol.hpp"

namespace finer::server {

class Transport : public OutputSink {
public:
  using Requests = std::vector<std::pair<RequestType, nlohmann::json>>;

  Transport(const ServerConfig &cfg, int rank);
  ~Transport();

  /// 握手前半：HELLO → INIT，拿到数据面地址。在重量级初始化（建图编译）
  /// **之前**调用：前端立刻看到引擎注册，且其 ready-timeout 的 READY 阶段
  /// 从这里重新计时——大图编译只占第二段预算，不占第一段。
  void handshake_begin();
  /// 握手后半：等所有 rank 建完图，再数据面连接 + ReadyResponse + READY。
  /// The ReadyResponse reports the runner's actual KV pool geometry.
  void handshake_complete(const cache::KVCacheInfo &kv);

  /// Requests for this step, identical on every rank. An empty batch is still
  /// a broadcast round, so followers never run ahead of rank0.
  Requests recv(int timeout_ms);
  /// rank0's sampled tokens become every rank's tokens: TopK ties need not
  /// resolve identically across ranks.
  void sync_tokens(std::vector<int64_t> &tokens);
  bool all_have_capacity(bool local) override {
    int value = local ? 1 : 0, result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return result != 0;
  }

  void send_outputs(const OutputBatch &batch);
  void send_utility(uint64_t call_id, const nlohmann::json &result);

private:
  bool leader() const {
    return rank_ == 0;
  }
  bool  recv_frames(std::vector<uint8_t> &batch, int timeout_ms);
  void *make_handshake_socket(const ReadyMessage &msg);
  void  send_buf(void *sock, const std::vector<uint8_t> &data) {
    if (leader())
      zmq_send(sock, data.data(), data.size(), 0);
  }

  const ServerConfig &cfg_;
  int                 rank_;
  void               *ctx_    = nullptr;
  void               *dealer_ = nullptr;
  void               *push_   = nullptr;
  std::string         input_addr_, output_addr_; // handshake_begin 存
};

// Implementation

namespace detail {
constexpr int kZmqTimeoutMs = 30000;
inline void   set_timeout(void *sock, int ms) {
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &ms, sizeof(ms));
  zmq_setsockopt(sock, ZMQ_SNDTIMEO, &ms, sizeof(ms));
}
inline void set_identity(void *sock, uint32_t engine_index) {
  // vLLM addresses engines by a 2-byte little-endian identity.
  const auto id = static_cast<uint16_t>(engine_index);
  zmq_setsockopt(sock, ZMQ_IDENTITY, &id, sizeof(id));
}
} // namespace detail

inline Transport::Transport(const ServerConfig &cfg, int rank)
    : cfg_(cfg), rank_(rank) {
  if (!leader())
    return;
  ctx_    = zmq_ctx_new();
  dealer_ = zmq_socket(ctx_, ZMQ_DEALER);
  push_   = zmq_socket(ctx_, ZMQ_PUSH);
  detail::set_identity(dealer_, cfg_.engine_index);
  detail::set_timeout(dealer_, detail::kZmqTimeoutMs);
  detail::set_timeout(push_, detail::kZmqTimeoutMs);
}

inline Transport::~Transport() {
  if (!leader())
    return;
  zmq_close(dealer_);
  zmq_close(push_);
  zmq_ctx_destroy(ctx_);
}

// One-shot DEALER to the handshake address, sends `msg`, returns the socket
// for the caller to recv on / close.
inline void *Transport::make_handshake_socket(const ReadyMessage &msg) {
  void *hs = zmq_socket(ctx_, ZMQ_DEALER);
  detail::set_identity(hs, cfg_.engine_index);
  detail::set_timeout(hs, detail::kZmqTimeoutMs);
  zmq_connect(hs, cfg_.handshake_address.c_str());
  send_buf(hs, encode_ready_message(msg));
  return hs;
}

// vLLM-compat ReadyResponse. world_size/data_parallel_size are fixed at 1
// (single engine, single-wave DP). Block geometry is the runner's real pool;
// dtype stays a compatibility value and does not configure model precision.
inline std::vector<uint8_t> encode_ready_response(
  const ServerConfig &cfg, const cache::KVCacheInfo &kv) {
  const auto tokens = kv.num_blocks * kv.block_size;
  return encode_msgpack({{"max_model_len", cfg.max_model_len},
    {"num_gpu_blocks", kv.num_blocks}, {"block_size", kv.block_size},
    {"dtype", "bfloat16"}, {"vllm_version", cfg.vllm_version},
    {"world_size", 1}, {"data_parallel_size", 1}, {"dp_stats_address", nullptr},
    {"kv_cache_size_tokens", tokens},
    {"kv_cache_max_concurrency",
      double(tokens) / double(std::max<int64_t>(cfg.max_model_len, 1))}});
}

inline void Transport::handshake_begin() {
  if (!leader())
    return;
  spdlog::info("[handshake] connecting to {} (engine_index={})",
    cfg_.handshake_address, cfg_.engine_index);

  // Phase 1: handshake DEALER → HELLO → recv INIT
  void *hs = make_handshake_socket({"HELLO", true, false});

  zmq_msg_t init;
  zmq_msg_init(&init);
  if (zmq_msg_recv(&init, hs, 0) == -1)
    throw std::runtime_error("Transport: failed to receive INIT");
  auto init_json = decode_msgpack(zmq_msg_data(&init), zmq_msg_size(&init));
  zmq_msg_close(&init);
  zmq_close(hs);

  auto addrs   = init_json["addresses"];
  input_addr_  = addrs["inputs"][0].get<std::string>();
  output_addr_ = addrs["outputs"][0].get<std::string>();
  spdlog::info("[handshake] INIT received: input={}, output={}", input_addr_,
    output_addr_);
}

inline void Transport::handshake_complete(const cache::KVCacheInfo &kv) {
  // READY covers the whole TP group, including ranks that compiled slower.
  MPI_Barrier(MPI_COMM_WORLD);
  if (!leader())
    return;
  // Phase 2: input DEALER → ReadyResponse
  zmq_connect(dealer_, input_addr_.c_str());
  send_buf(dealer_, encode_ready_response(cfg_, kv));

  // Phase 3: output PUSH
  zmq_connect(push_, output_addr_.c_str());

  // Phase 4: READY via handshake
  zmq_close(make_handshake_socket({"READY", true, false}));

  // Switch input to non-blocking
  int zero = 0;
  zmq_setsockopt(dealer_, ZMQ_RCVTIMEO, &zero, sizeof(zero));
  spdlog::info("[handshake] READY sent, entering main loop");
}

// Appends one [size0, size1, frame0, frame1] record; MessagePack stays as
// received so followers decode exactly the bytes rank0 saw.
inline bool Transport::recv_frames(
  std::vector<uint8_t> &batch, int timeout_ms) {
  zmq_pollitem_t items[] = {{dealer_, 0, ZMQ_POLLIN, 0}};
  const int      polled  = zmq_poll(items, 1, timeout_ms);
  if (polled < 0 && zmq_errno() != EINTR) // EINTR: SIGTERM shutdown
    spdlog::warn("[transport] zmq_poll failed: {}", zmq_strerror(zmq_errno()));
  if (polled <= 0 || !(items[0].revents & ZMQ_POLLIN))
    return false;

  zmq_msg_t frames[2];
  zmq_msg_init(&frames[0]);
  zmq_msg_init(&frames[1]);
  const bool ok = zmq_msg_recv(&frames[0], dealer_, 0) != -1
    && zmq_msg_more(&frames[0]) && zmq_msg_recv(&frames[1], dealer_, 0) != -1;
  if (ok) {
    const uint64_t sizes[] = {
      zmq_msg_size(&frames[0]), zmq_msg_size(&frames[1])};
    const auto *s = reinterpret_cast<const uint8_t *>(sizes);
    batch.insert(batch.end(), s, s + sizeof(sizes));
    for (auto &frame : frames) {
      const auto *data = static_cast<const uint8_t *>(zmq_msg_data(&frame));
      batch.insert(batch.end(), data, data + zmq_msg_size(&frame));
    }
  } else if (zmq_errno() != EINTR) {
    spdlog::warn(
      "[transport] bad request frames: {}", zmq_strerror(zmq_errno()));
  }
  zmq_msg_close(&frames[0]);
  zmq_msg_close(&frames[1]);
  return ok;
}

inline Transport::Requests Transport::recv(int timeout_ms) {
  std::vector<uint8_t> bytes;
  // Wait only for the first message, then drain what is already queued.
  while (leader() && recv_frames(bytes, timeout_ms))
    timeout_ms = 0;

  // One step's requests are assumed to stay well below 2 GiB.
  int size = static_cast<int>(bytes.size());
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  bytes.resize(size);
  MPI_Bcast(bytes.data(), size, MPI_BYTE, 0, MPI_COMM_WORLD);

  Requests requests;
  for (size_t offset = 0; offset < bytes.size();) {
    uint64_t sizes[2];
    std::memcpy(sizes, bytes.data() + offset, sizeof(sizes));
    const auto *frame0 = bytes.data() + offset + sizeof(sizes);
    requests.push_back(
      decode_request_message(frame0, sizes[0], frame0 + sizes[0], sizes[1]));
    offset += sizeof(sizes) + sizes[0] + sizes[1];
  }
  return requests;
}

inline void Transport::sync_tokens(std::vector<int64_t> &tokens) {
  MPI_Bcast(tokens.data(), static_cast<int>(tokens.size()), MPI_INT64_T, 0,
    MPI_COMM_WORLD);
}

inline void Transport::send_outputs(const OutputBatch &batch) {
  if (batch.outputs.empty() && batch.finished_requests.empty()
    && !batch.scheduler_stats.has_value())
    return;
  send_buf(push_, encode_outputs(batch));
}

inline void Transport::send_utility(
  uint64_t call_id, const nlohmann::json &result) {
  send_buf(push_, encode_utility_response(call_id, result));
}

} // namespace finer::server
