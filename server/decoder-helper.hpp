#pragma once

#include "arch_config.hpp"
#include "channel.hpp"
#include "common_def.hpp"
#include "zmq.hpp"

// inject into decoder to provide communication with router via zmq
struct DecoderHelper {
protected:
  // zeromq mailbox stuff
  zmq::context_t zmq_ctx;
  zmq::socket_t  zmq_mailbox;
  zmq::socket_t  zmq_router;

public:
  channel::mpsc<ipcm::DecoderRequest> request_buffer;

  void zmq_init(const RouterConfig &router, const DecoderConfig &config) {
    zmq_ctx     = zmq::context_t(1);
    zmq_mailbox = zmq::socket_t(zmq_ctx, zmq::socket_type::pull);
    zmq_mailbox.bind(config.mailbox_addr);
    zmq_router = zmq::socket_t(zmq_ctx, zmq::socket_type::push);
    zmq_router.connect(router.mailbox_addr);
  }

  // blocking function, should be run in a separate thread
  void start_recv() {
    while (true) {
      zmq::message_t       zmq_msg;
      auto                 _ = zmq_mailbox.recv(zmq_msg, zmq::recv_flags::none);
      ipcm::DecoderRequest req =
          ipcm::DecoderRequest::from_message(zmq_msg.to_string_view());
      request_buffer.emplace(std::move(req));
    }
  }

  void send_update(const ipcm::DecoderUpdate &update) {
    zmq::message_t zmq_msg(update.to_message());
    zmq_router.send(zmq_msg, zmq::send_flags::none);
  }
};