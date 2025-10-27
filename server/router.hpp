#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "arch_config.hpp"
#include "channel.hpp"
#include "common_def.hpp"
#include "httplib.h"
#include "json.hpp"
#include "zmq.hpp"

namespace router {

struct RunningTask {
  int                           id;
  openai::ChatCompletionRequest req;
  // channel to send updates back to HTTP handler
  // using unique_ptr to avoid moving mutex inside mpsc
  std::unique_ptr<channel::mpsc<ipcm::DecoderUpdate>> channel;
};

struct RunningTaskTable : protected std::unordered_map<int, RunningTask> {
  std::mutex mutex;
  int        request_id = 0;

public:
  struct RunningTaskBuilder {
    openai::ChatCompletionRequest req;
  };
  RunningTask &emplace(RunningTaskBuilder &&val) {
    std::unique_lock<std::mutex> lock(mutex);

    int id             = request_id++;
    auto [it, success] = std::unordered_map<int, RunningTask>::emplace(
        id,
        RunningTask{
            .id      = id,
            .req     = std::move(val.req),
            .channel = std::make_unique<channel::mpsc<ipcm::DecoderUpdate>>(),
        });
    assert(success);
    return it->second;
  }

  void update(ipcm::DecoderUpdate &&update) {
    std::unique_lock<std::mutex> lock(mutex);

    auto &channel =
        std::unordered_map<int, RunningTask>::find(update.id)->second.channel;
    switch (update.type) {
    case ipcm::DecoderUpdate::Update:
      channel->emplace(std::move(update));
      break;
    case ipcm::DecoderUpdate::Finish:
      channel->close();
      break;
    }
  }
};

struct Decoder {
protected:
  zmq::socket_t               socket;
  std::unique_ptr<std::mutex> mutex;
  // other workload info...

public:
  Decoder(zmq::context_t &context, const DecoderConfig &config) {
    socket = zmq::socket_t(context, zmq::socket_type::push);
    socket.connect(config.mailbox_addr);
    mutex = std::make_unique<std::mutex>();
  }

  void dispatch_task(const RunningTask &task) {
    std::unique_lock<std::mutex> lock(*mutex);
    // send task to decoder via zmq
    ipcm::DecoderRequest msg({
        .id    = task.id,
        .model = task.req.model,
        .input = task.req.messages.back().content,
    });
    socket.send(zmq::message_t(msg.to_message()), zmq::send_flags::none);
  }
};

struct DecoderPool {
protected:
  std::vector<Decoder> workers;

  // simple round-robin scheduling
  struct {
    size_t     current = 0;
    size_t     total   = 0;
    std::mutex mutex;

    size_t get() {
      std::unique_lock<std::mutex> lock(mutex);

      size_t res = current;
      current    = (current + 1) % total;
      return res;
    }
  } next_worker;

public:
  template <class... Args>
  void register_worker(Args &&...args) {
    workers.emplace_back(std::forward<Args>(args)...);
    next_worker.total = workers.size();
  }

  void dispatch_task(const RunningTask &task) {
    workers[next_worker.get()].dispatch_task(task);
  }
};

static void start(const ArchConfig &config) {
  // PART: mailbox
  zmq::context_t zmq_ctx(1);
  zmq::socket_t  mailbox_socket(zmq_ctx, zmq::socket_type::pull);
  mailbox_socket.bind(config.router.mailbox_addr);

  // PART: running task table
  RunningTaskTable running_task;

  // PART: Decoder Pool
  DecoderPool decoder_pool;
  for (const auto &worker_config : config.workers) {
    decoder_pool.register_worker(zmq_ctx, worker_config);
  }

  // PART: Mailbox Recv Thread
  std::thread mailbox_thread([&mailbox_socket, &running_task]() {
    while (true) {
      zmq::message_t msg;
      auto           _ = mailbox_socket.recv(msg, zmq::recv_flags::none);
      auto update = ipcm::DecoderUpdate::from_message(msg.to_string_view());
      // printf("receive decoder update for request %d: %s\n", update.id,
      //        update.content.c_str());

      running_task.update(std::move(update));
    }
  });

  // PART: http server
  auto svr = httplib::Server();
  // set thread pool with size `config.router.num_http_threads`
  svr.new_task_queue = [num_http_threads = config.router.num_http_threads] {
    return new httplib::ThreadPool(num_http_threads);
  };

  auto handle_response_unstreamed = [](RunningTask       &task,
                                       httplib::Response &res) {
    openai::ChatCompletionResponse resp{
        .choices = {(openai::ChatCompletionResponse::Choice){
            .message = std::optional<openai::ChatCompletionResponse::Message>({
                .role    = "assistant",
                .content = "",
            }),
        }}};
    auto &resp_content = resp.choices[0].message->content;
    for (auto update : *task.channel) {
      resp_content += update.content;
    }
    res.set_content(resp.to_json().dump(), "application/json");
  };
  auto handle_response_streamed = [](RunningTask       &task,
                                     httplib::Response &res) {
    res.set_content_provider(
        "text/event-stream",
        [&task](size_t /*_offset*/, httplib::DataSink &sink) -> bool {
          auto sink_write = [&sink](const std::string &data) {
            std::string ev_data = "data: " + data + "\n\n";
            sink.write(ev_data.data(), ev_data.size());
          };

          auto update = task.channel->pop();
          if (!update.has_value()) {
            sink_write("[DONE]");
            return false;
          } else {
            openai::ChatCompletionResponse resp{
                .choices = {(openai::ChatCompletionResponse::Choice){
                    .delta =
                        std::optional((openai::ChatCompletionResponse::Delta){
                            .content = update->content,
                        }),
                }}};
            sink_write(resp.to_json().dump());
            return true;
          }
        },
        [](bool _) {
          // TODO: send this to decoder to cancel the task
        });
  };

  // clang-format off
  svr.Post("/v1/chat/completions", 
    [&](const httplib::Request &req_raw, httplib::Response &res) {
    printf("recv request: %s\n", req_raw.body.c_str());
    auto req = openai::ChatCompletionRequest::from_json(nlohmann::json::parse(req_raw.body));

    // register request in task table
    auto &task = running_task.emplace({
        .req = std::move(req),
    });

    // dispatch task to decoder pool
    decoder_pool.dispatch_task(task);

    if (req.stream) {
      handle_response_streamed(task, res);
    } else {
      handle_response_unstreamed(task, res);
    }
  });
  // clang-format on

  svr.listen(config.router.input_addr, config.router.input_port);
  mailbox_thread.join();
}

} // namespace router
