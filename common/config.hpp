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
    std::string model_path;
    std::string system_prompt = R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
User:)";

    ModelConfig(const YAML::Node & config) {
        model_path    = config["model_path"].as<std::string>();
        system_prompt = config["system_prompt"].as<std::string>(system_prompt);
    }
};

struct BackendConfig {
    // number of parallel sequences to process
    uint32_t n_parallel = 1;

    // context length (for kv cache)
    uint32_t n_context = 512;

    // number of tokens to process in a batch
    uint32_t n_batch = 512;

    // number of threads to use for computation, for ggml-cpu
    uint32_t n_threads = 64;

    // defragmentation threshold, 0 means no defragmentation
    float defrag_thold = 0;

    bool debug = false;

    BackendConfig(const YAML::Node & config) {
        n_parallel   = config["n_parallel"].as<uint32_t>(n_parallel);
        n_context    = config["n_context"].as<uint32_t>(n_context);
        n_batch      = config["n_batch"].as<uint32_t>(n_batch);
        n_threads    = config["n_threads"].as<uint32_t>(n_threads);
        defrag_thold = config["defrag_thold"].as<float>(defrag_thold);
        debug        = config["debug"].as<bool>(debug);
    }
};

struct Config {
    ModelConfig   model;
    BackendConfig backend;
    ServerConfig  server;

    Config(const YAML::Node & config) : model(config["model"]), backend(config["backend"]), server(config["server"]) {}
};
