#pragma once

#include "yaml-cpp/yaml.h"

#include <string>
#include <vector>

struct RouterConfig {
  std::string input_addr;
  int         input_port;

  int num_http_threads;

  std::string mailbox_addr;

  // constructor from YAML::Node
  RouterConfig(const YAML::Node &config) {
    input_addr       = config["listen_addr"].as<std::string>();
    input_port       = config["listen_port"].as<int>();
    num_http_threads = config["num_http_threads"].as<int>();
    mailbox_addr     = config["mailbox_addr"].as<std::string>();
  }
};

struct DecoderConfig {
  std::string mailbox_addr;

  DecoderConfig(const YAML::Node &config) {
    mailbox_addr = config["mailbox_addr"].as<std::string>();
  }
};

struct ArchConfig {
  RouterConfig               router;
  std::vector<DecoderConfig> decoders;

  ArchConfig(const YAML::Node &config) : router(config["router"]) {
    for (const auto &worker_node : config["decoders"]) {
      decoders.emplace_back(worker_node);
    }
  }
};