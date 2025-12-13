#pragma once

#include <optional>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

struct RouterConfig {
    std::string input_addr;
    int         input_port;

    int num_http_threads;

    std::string mailbox_addr;

    // constructor from YAML::Node
    RouterConfig(const YAML::Node & config) {
        input_addr       = config["input_addr"].as<std::string>();
        input_port       = config["input_port"].as<int>();
        num_http_threads = config["num_http_threads"].as<int>();
        mailbox_addr     = config["mailbox_addr"].as<std::string>();
    }
};

struct DecoderConfig {
    std::string mailbox_addr;

    DecoderConfig(const YAML::Node & config) { mailbox_addr = config["mailbox_addr"].as<std::string>(); }
};

struct ServerConfig {
    enum Mode { Standalone, Server } mode;

    std::optional<RouterConfig> router;
    std::vector<DecoderConfig>  decoders;

    ServerConfig(const YAML::Node & config) {
        std::string mode_str = config["mode"].as<std::string>("standalone");
        if (mode_str == "standalone") {
            mode = Standalone;
        } else if (mode_str == "server") {
            mode = Server;
        } else {
            throw std::runtime_error("Unknown server mode: " + mode_str);
        }

        if (mode == Server) {
            router = RouterConfig(config["router"]);
            for (const auto & worker_node : config["decoders"]) {
                decoders.emplace_back(worker_node);
            }
        }
    }
};

struct ModelConfig {
    ModelConfig(const YAML::Node & config) {}
};

struct Config {
    ModelConfig  model;
    ServerConfig server;

    Config(const YAML::Node & config) : model(config["model"]), server(config["server"]) {}
};
