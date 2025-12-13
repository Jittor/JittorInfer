#include "common_def.hpp"
#include "decoder-helper.hpp"
#include <chrono>
#include <cstdio>
#include <thread>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  assert(argc >= 2 && "Usage: server <config.yaml>");
  Config config{YAML::LoadFile(argv[1])};

  printf("Decoder Worker %d starting, "
         "binding to mailbox at %s, "
         "connecting to router at %s\n",
         mpi_rank, config.server.decoders[mpi_rank].mailbox_addr.c_str(),
         config.server.router->mailbox_addr.c_str());
  DecoderHelper helper;
  helper.init(config, mpi_rank);
  std::thread recv_thread([&helper]() { helper.start_recv(); });

  for (auto req : helper.request_buffer) {
    // pretend to process requests
    printf("[%d] receive request: model: %s, input: %s\n", mpi_rank,
           req.model.c_str(), req.input.c_str());
    for (int i = 0; i < 10; i++) {
      ipcm::DecoderUpdate update{
          .type    = ipcm::DecoderUpdate::Update,
          .id      = req.id,
          .content = "#" + std::to_string(i),
      };
      helper.send_update(update);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    helper.send_update({.type = ipcm::DecoderUpdate::Finish, .id = req.id});
  }

  recv_thread.join();
  return 0;
}