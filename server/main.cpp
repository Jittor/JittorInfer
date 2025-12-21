#include "../common/config.hpp"
#include "router.hpp"

#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
  assert(argc >= 2 && "Usage: server <config.yaml>");
  Config config{YAML::LoadFile(argv[1])};
  assert(config.server.mode == Config::Server &&
         "This executable only runs in server mode");

  // dump config info:
  std::cout << "Router listening on " << config.server.router->input_addr << ":"
            << config.server.router->input_port << "\n";
  std::cout << "Number of HTTP threads: "
            << config.server.router->num_http_threads << "\n";
  std::cout << "Router mailbox address: " << config.server.router->mailbox_addr
            << "\n";
  std::cout << "Decoders:\n";
  for (const auto &decoder : config.server.decoders) {
    std::cout << "  - " << decoder.mailbox_addr << "\n";
  }

  router::start(config.server);

  return 0;
}