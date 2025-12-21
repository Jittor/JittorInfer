#pragma once

#include "../common/config.hpp"
#include "channel.hpp"
#include "common_def.hpp"
#include "zmq.hpp"
#include <cstdio>
#include <thread>

// inject into decoder to provide communication with router via zmq
struct DecoderHelper {
protected:
  // zeromq mailbox stuff
  zmq::context_t zmq_ctx;
  zmq::socket_t  zmq_mailbox;
  zmq::socket_t  zmq_router;

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

  // return the last index of character that can form a valid string
  // if the last character is potentially cut in half, return the index before
  // the cut if validate_utf8(text) == text.size(), then the whole text is valid
  // utf8
  static size_t _validate_utf8(const std::string &text) {
    size_t len = text.size();
    if (len == 0) {
      return 0;
    }

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
      unsigned char c = text[len - i];
      // Check for start of a multi-byte sequence from the end
      if ((c & 0xE0) == 0xC0) {
        // 2-byte character start: 110xxxxx
        // Needs at least 2 bytes
        if (i < 2) {
          return len - i;
        }
      } else if ((c & 0xF0) == 0xE0) {
        // 3-byte character start: 1110xxxx
        // Needs at least 3 bytes
        if (i < 3) {
          return len - i;
        }
      } else if ((c & 0xF8) == 0xF0) {
        // 4-byte character start: 11110xxx
        // Needs at least 4 bytes
        if (i < 4) {
          return len - i;
        }
      }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
  }

public:
  channel::mpsc<ipcm::DecoderRequest> request_buffer;
  ServerConfig::Mode                  mode;

  void init(const Config &config, int mpi_rank) {
    mode = config.server.mode;
    if (mode == ServerConfig::Standalone) {
      printf("Decoder initialized in standalone mode\n");
    } else {
      zmq_ctx     = zmq::context_t(1);
      zmq_mailbox = zmq::socket_t(zmq_ctx, zmq::socket_type::pull);
      zmq_mailbox.bind(config.server.decoders[mpi_rank].mailbox_addr);
      zmq_router = zmq::socket_t(zmq_ctx, zmq::socket_type::push);
      zmq_router.connect(config.server.router->mailbox_addr);

      // printf mpi rank <-> pid for profiling and debugging
      int pid = static_cast<int>(getpid());
      printf("Decoder rank %d started with PID %d\n", mpi_rank, pid);

      std::thread([this]() { this->start_recv(); }).detach();
    }
  }

  void send_update(const ipcm::DecoderUpdate &update) {
    if (mode == ServerConfig::Standalone) {
      auto msg = update.to_message();
      printf("Decoder update: %s\n", msg.c_str());
    } else {
      zmq::message_t zmq_msg(update.to_message());
      zmq_router.send(zmq_msg, zmq::send_flags::none);
    }
  }

  static std::string retrieve_valid_utf8(std::string &text) {
    size_t      valid_len = _validate_utf8(text);
    std::string result    = text.substr(0, valid_len);
    text                  = text.substr(valid_len);

    // if result still start with cut-off character, remove it
    for (char &c : result) {
      if ((c & 0b11000000) == 0b10000000) {
        c = ' '; // replace invalid byte with space
      } else {
        break;
      }
    }
    return result;
  }
};