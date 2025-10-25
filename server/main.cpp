#include "arch_config.hpp"
#include "router.hpp"

#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
  assert(argc >= 2 && "Usage: server <config.yaml>");
  ArchConfig config{YAML::LoadFile(argv[1])};

  // dump config info:
  std::cout << "Router listening on " << config.router.input_addr << ":"
            << config.router.input_port << "\n";
  std::cout << "Number of HTTP threads: " << config.router.num_http_threads
            << "\n";
  std::cout << "Router mailbox address: " << config.router.mailbox_addr << "\n";
  std::cout << "Workers:\n";
  for (const auto &worker : config.workers) {
    std::cout << "  - " << worker.mailbox_addr << "\n";
  }
  
  router::start(config);
  
  return 0;
}