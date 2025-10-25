#pragma once

#include "json.hpp"
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ipc messages between router and decoder
namespace ipcm {

struct DecoderRequest {
  int         id;
  std::string model;
  std::string input;

  static DecoderRequest from_message(const std::string_view &msg) {
    auto json = nlohmann::json::parse(msg);
    return DecoderRequest{
        .id    = json["id"],
        .model = json["model"],
        .input = json["input"],
    };
  }

  std::string to_message() const {
    return nlohmann::json{
        {"id", id},
        {"model", model},
        {"input", input},
    }
        .dump();
  }
};

struct DecoderUpdate {
  enum {
    Update,
    Finish,
  } type;

  int         id;
  std::string content;

  static DecoderUpdate from_message(const std::string_view &msg) {
    auto json = nlohmann::json::parse(msg);
    return DecoderUpdate{
        .type    = json["type"],
        .id      = json["id"],
        .content = json["content"],
    };
  }

  std::string to_message() const {
    return nlohmann::json{{"type", type}, {"id", id}, {"content", content}}
        .dump();
  }
};

} // namespace ipcm

// namespace for OpenAI compatible API
namespace openai {

// These fields are collected from inference-benchmark, the benckmark tool we're
// now using.
// TODO: add more fields according to OpenAI API spec
struct ChatCompletionRequest {
  struct Message {
    std::string role; // "system", "user", "assistant"
    std::string content;
  };

  std::string          model;
  std::vector<Message> messages;
  int                  max_tokens;
  bool                 stream;
  // std::vector<std::string> stop;
  float temperature;

  static ChatCompletionRequest from_json(const nlohmann::json &raw) {
    ChatCompletionRequest req;
    req.model       = raw["model"];
    req.max_tokens  = raw.value("max_tokens", -1);
    req.stream      = raw.value("stream", false);
    req.temperature = raw.value("temperature", 0.0f);
    for (const auto &msg : raw["messages"]) {
      req.messages.push_back({.role = msg["role"], .content = msg["content"]});
    }
    return req;
  }
};

struct ChatCompletionResponse {
  // full response
  struct Message {
    std::string role;
    std::string content;
  };
  // partial response
  struct Delta {
    std::string content;
  };

  struct Choice {
    std::optional<Message>     message;
    std::optional<std::string> finish_reason;
    std::optional<Delta>       delta;
  };

  std::vector<Choice> choices;

  nlohmann::json to_json() const {
    nlohmann::json res;
    res["choices"] = nlohmann::json::array();
    for (const auto &choice : choices) {
      nlohmann::json choice_json;
      if (choice.message.has_value()) {
        choice_json["message"] = {
            {"role", choice.message->role},
            {"content", choice.message->content},
        };
      }
      if (choice.finish_reason.has_value()) {
        choice_json["finish_reason"] = choice.finish_reason.value();
      }
      if (choice.delta.has_value()) {
        choice_json["delta"] = {
            {"content", choice.delta->content},
        };
      }
      res["choices"].emplace_back(std::move(choice_json));
    }
    return res;
  }
};

} // namespace openai