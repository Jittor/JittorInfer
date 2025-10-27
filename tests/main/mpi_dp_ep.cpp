// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "../server/decoder-helper.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common_def.hpp"
#include "common_local.h"
#include "ggml.h"
#include "llama.h"
#include "sampling_local.h"

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

static std::string k_system =
    R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
User:)";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
};

struct Client {
  ~Client() {
    if (smpl) {
      common_sampler_local_free(smpl);
    }
  }

  // request id from outside
  llama_seq_id req_id     = -1;
  bool         is_running = false;

  int32_t ith_client = 0;
  int32_t ith_batch  = -1;

  llama_token last_token;

  int64_t t_start_prompt;
  int64_t t_start_gen;

  int32_t n_prompt  = 0;
  int32_t n_decoded = 0;

  std::string              input_string;
  std::vector<llama_token> input_tokens;
  std::string              output_string;

  // llms usually split a whole utf-8 character into multiple tokens,
  // we need to buffer the last incomplete character here
  std::string output_buffer;

  struct common_sampler_local *smpl = nullptr;

  struct ResetParam {
    std::string              input_string;
    std::vector<llama_token> input_tokens;
    int                      req_id = -1;
  };

  void reset(ResetParam &&param) {
    req_id     = param.req_id;
    is_running = true;

    t_start_prompt = ggml_time_us();

    GGML_ASSERT(!param.input_tokens.empty());
    input_string  = std::move(param.input_string);
    input_tokens  = std::move(param.input_tokens);
    last_token    = input_tokens.back();
    output_string = "";
    output_buffer = "";
    n_prompt      = input_tokens.size();
    n_decoded     = 0;

    common_sampler_reset(smpl);
  }
};

static struct DefaultMiniParams {
  int main_gpu     = 1;
  int n_gpu_layers = 99;

  // how split tensors should be distributed across GPUs
  float tensor_split[128] = {0};

  std::string model = "/root/data/DeepSeek-V2-Lite-Chat-f16.gguf";

  uint32_t n_ctx     = 512; // context size
  uint32_t n_threads = 64;  // number of threads to use for computation
  uint32_t n_threads_batch =
      64; // number of threads to use for batch processing

  float defrag_thold = 0.1f; // defragmentation threshold
  bool  no_perf      = true; // disable performance metrics
  std::vector<common_adapter_lora_info>
      lora_adapters; // lora adapter path with user defined scale

  int32_t n_batch = 512; // logical batch size for prompt processing
  bool    enable_chat_template               = true;
  bool    escape                             = true;
  common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;

  // parallel test configs
  int     n_parallel    = 16;
  bool    dump_kv_cache = false;
  int32_t n_predict     = 510; // new tokens to predict

} default_mini_params;

static llama_model_params common_model_params_to_llama_local() {
  auto mparams = llama_model_default_params();

  mparams.main_gpu      = default_mini_params.main_gpu;
  mparams.split_mode    = LLAMA_SPLIT_MODE_ROW;
  mparams.tensor_split  = default_mini_params.tensor_split;
  mparams.use_mmap      = true;
  mparams.use_mlock     = false;
  mparams.check_tensors = false;
  mparams.n_gpu_layers  = default_mini_params.n_gpu_layers;
  // 开启张量并行
  mparams.enable_tensor_parallel = true;
  mparams.enable_expert_parallel = true;
  mparams.enable_data_parallel   = true;
  mparams.enable_fused_moe       = true;
  mparams.enable_mpi             = true;

  if (mparams.enable_mpi) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mparams.tp_id);
    MPI_Comm_size(MPI_COMM_WORLD, &mparams.num_parallel);
  }

  mparams.kv_overrides = NULL;
  return mparams;
}

static llama_context_params common_context_params_to_llama_local() {
  auto cparams = llama_context_default_params();

  cparams.n_ctx   = default_mini_params.n_ctx * default_mini_params.n_parallel;
  cparams.n_batch = default_mini_params.n_batch;
  cparams.n_threads       = default_mini_params.n_threads;
  cparams.n_threads_batch = default_mini_params.n_threads_batch;
  cparams.defrag_thold    = default_mini_params.defrag_thold;
  cparams.no_perf         = default_mini_params.no_perf;
  cparams.presample_count = 80;

  return cparams;
}

static common_init_result common_init_from_params_local() {
  common_init_result iparams;

  auto mparams = common_model_params_to_llama_local();

  llama_model *model =
      llama_model_load_from_file(default_mini_params.model.c_str(), mparams);
  assert(model != NULL);

  const llama_vocab *vocab = llama_model_get_vocab(model);

  auto cparams = common_context_params_to_llama_local();

  llama_context *lctx = llama_init_from_model(model, cparams);

  if (lctx == NULL) {
    std::cerr << __func__ << ": failed to create context with model '"
              << default_mini_params.model << "'\n";
    llama_model_free(model);
    return iparams;
  }

  GGML_ASSERT(default_mini_params.lora_adapters.empty());

  llama_clear_adapter_lora(lctx);

  // create warmup sequence with BOS and EOS tokens
  std::vector<llama_token> tmp;
  llama_token              bos = llama_vocab_bos(vocab);
  llama_token              eos = llama_vocab_eos(vocab);
  if (bos != LLAMA_TOKEN_NULL) {
    tmp.push_back(bos);
  }
  if (eos != LLAMA_TOKEN_NULL) {
    tmp.push_back(eos);
  }
  if (tmp.empty()) {
    tmp.push_back(0);
  }

  GGML_ASSERT(!llama_model_has_encoder(model));

  if (llama_model_has_decoder(model)) {
    llama_decode(lctx,
                 llama_batch_get_one(
                     tmp.data(),
                     std::min(tmp.size(), (size_t)default_mini_params.n_batch)),
                 true);
  }

  llama_kv_cache_clear(lctx);
  llama_synchronize(lctx);
  llama_perf_context_reset(lctx);

  iparams.model.reset(model);
  iparams.context.reset(lctx);

  return iparams;
}

int main(int argc, char **argv) {
  int mpi_rank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // load decoder-helper
  GGML_ASSERT(argc >= 2 && "Usage: mpi_dp_ep <arch_config.yml>");
  ArchConfig    config{YAML::LoadFile(argv[1])};
  DecoderHelper decoder_helper;
  decoder_helper.zmq_init(config.router, config.workers[mpi_rank]);
  std::thread recv_thread([&decoder_helper]() { decoder_helper.start_recv(); });

  // number of simultaneous "clients" to simulate
  const int32_t n_clients = default_mini_params.n_parallel;

  // dedicate one sequence to the system prompt
  // so as to reserve a kv cache slot for it
  default_mini_params.n_parallel += 1;

  const bool dump_kv_cache = default_mini_params.dump_kv_cache;

  // init llama.cpp
  llama_backend_init();
  // llama_numa_init(params.numa);

  // load the target model
  common_init_result llama_init = common_init_from_params_local();

  llama_model   *model = llama_init.model.get();
  llama_context *ctx   = llama_init.context.get();

  const llama_vocab *vocab = llama_model_get_vocab(model);

  const int n_ctx = llama_n_ctx(ctx);

  common_params_local_sampling sparams;
  std::vector<Client>          clients(n_clients);
  for (size_t i = 0; i < clients.size(); ++i) {
    auto &client      = clients[i];
    client.ith_client = i;
    client.smpl       = common_sampler_local_init(model, sparams);
  }

  std::vector<llama_token> tokens_system;
  tokens_system                 = common_tokenize(ctx, k_system, true);
  const int32_t n_tokens_system = tokens_system.size();

  // the max batch size is as large as the context to handle cases where we get
  // very long input prompt from multiple users. regardless of the size, the
  // main loop will chunk the batch into a maximum of params.n_batch tokens at a
  // time
  llama_batch batch =
      llama_batch_init(n_ctx, 0, default_mini_params.n_parallel);

  // int32_t n_total_prompt = 0;
  // int32_t n_total_gen    = 0;
  int32_t n_cache_miss = 0;

  struct llama_kv_cache_view kvc_view =
      llama_kv_cache_view_init(ctx, n_clients);

  printf("%s: n_parallel = %d, , system tokens = %d\n", __func__, n_clients,
         n_tokens_system);

  { // prefill system prompt
    printf("%s: Evaluating the system prompt ...\n", __func__);

    for (int32_t i = 0; i < n_tokens_system; ++i) {
      common_batch_add(batch, tokens_system[i], i, {0}, false);
    }

    if (llama_decode(ctx, batch, true) != 0) {
      fprintf(stderr, "%s: llama_decode() failed\n", __func__);
      return 1;
    }

    // assign the system KV cache to all parallel sequences
    for (int32_t i = 1; i <= n_clients; ++i) {
      llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
    }

    printf("\n");
  }

  printf("System prompt prefilled, start processing requests ...\n\n");

  while (true) {

    if (dump_kv_cache) {
      llama_kv_cache_view_update(ctx, &kvc_view);
      common_kv_cache_dump_view_seqs(kvc_view, 40);
    }

    common_batch_clear(batch);

    // decode any currently ongoing sequences
    // 这已经是decode阶段了，n_batch数量等于待处理序列数量.prefill阶段以及全部处理结束seq_id
    // = -1
    for (auto &client : clients) {
      if (client.is_running) {
        client.ith_batch = batch.n_tokens;
        common_batch_add(batch, client.last_token,
                         n_tokens_system + client.n_prompt + client.n_decoded,
                         {client.ith_client + 1}, true);
        client.n_decoded += 1;
      }
    }

    if (batch.n_tokens == 0) {
      // all sequences have ended - clear the entire KV cache
      for (int i = 1; i <= n_clients; ++i) {
        llama_kv_cache_seq_rm(ctx, i, -1, -1);
        // but keep the system prompt
        llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
      }
      // printf("%s: clearing the KV cache\n", __func__);
    }

    // insert new sequences for decoding
    // 这个阶段是prefill，结束后将所有client的tokens连成一个batch，唯一一个n_tokens很长的阶段，decode阶段g_seq_id
    // == n_seq
    for (auto &client : clients) {
      if (!client.is_running) {
        auto req = decoder_helper.request_buffer.try_pop();

        if (req.has_value()) {
          std::vector<llama_token> prompt =
              common_tokenize(ctx, req->input + "\nAssistant:", false);
          client.reset({
              .input_string = std::move(req->input),
              .input_tokens = std::move(prompt),
              .req_id       = req->id,
          });

          client.ith_batch = batch.n_tokens - 1;
          for (size_t i = 0; i < client.input_tokens.size(); ++i) {
            common_batch_add(batch, client.input_tokens[i], i + n_tokens_system,
                             {client.ith_client + 1}, false);
          }
          batch.logits[batch.n_tokens - 1] = true;
        }
      }
    }

    if (batch.n_tokens == 0) {
      common_batch_add(batch, 0, n_tokens_system, {1}, false);
    }

    { // print fake batch percentage
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == 0) {
        printf("\033[2J\033[1;1H");
      }
      MPI_Barrier(MPI_COMM_WORLD);

      int n_running = 0;
      for (auto &client : clients) {
        n_running += client.is_running;
      }
      const float running_pct = 100.0f * n_running / clients.size();
      printf("\033[31m[%d]: batch tokens = %4d, running clients = %4d "
             "(%.2f%%)\033[0m\n",
             mpi_rank, batch.n_tokens, n_running, running_pct);
      fflush(stdout);

      MPI_Barrier(MPI_COMM_WORLD);
    }

    // empty run seems not working
    GGML_ASSERT(batch.n_tokens != 0);

    // process in chunks of params.n_batch
    int32_t n_batch = default_mini_params.n_batch;

    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
      const int32_t n_tokens = std::min(n_batch, (batch.n_tokens - i));

      llama_batch batch_view = {
          n_tokens,           batch.token + i,  nullptr,          batch.pos + i,
          batch.n_seq_id + i, batch.seq_id + i, batch.logits + i,
      };

      const int ret = llama_decode(ctx, batch_view, true);
      if (ret != 0) {
        if (n_batch == 1 || ret < 0) {
          // if you get here, it means the KV cache is full - try increasing it
          // via the context size
          fprintf(stderr,
                  "%s : failed to decode the batch, n_batch = %d, ret = %d\n",
                  __func__, n_batch, ret);
          return 1;
        }

        fprintf(stderr,
                "%s : failed to decode the batch, retrying with n_batch = %d\n",
                __func__, n_batch / 2);

        n_cache_miss += 1;

        // retry with half the batch size to try to find a free slot in the KV
        // cache
        n_batch /= 2;
        i -= n_batch;

        continue;
      }

      for (auto &client : clients) {
        if (client.ith_batch < (int)i ||
            client.ith_batch >= (int)(i + n_tokens)) {
          continue;
        }
        if (!client.is_running) {
          continue;
        }

        const llama_token sampled_token =
            common_sampler_sample(client.smpl, ctx, client.ith_batch - i);
        common_sampler_accept(client.smpl, sampled_token, true);

        if (client.n_decoded == 1) {
          client.t_start_gen = ggml_time_us();
        }

        const std::string sampled_string =
            common_token_to_piece(ctx, sampled_token);
        client.output_string += sampled_string;
        client.last_token = sampled_token;

        client.output_buffer += sampled_string;
        std::string output_sent =
            DecoderHelper::retrieve_valid_utf8(client.output_buffer);

        if (!output_sent.empty()) {
          decoder_helper.send_update({
              .type    = ipcm::DecoderUpdate::Update,
              .id      = client.req_id,
              .content = output_sent,
          });
        }

        auto is_finish = [&] {
          if (client.n_decoded > 2) {
            // eog
            if (llama_vocab_is_eog(vocab, client.last_token)) {
              return true;
            }
            // max tokens reached
            if (default_mini_params.n_predict > 0) {
              if (client.n_decoded + client.n_prompt >=
                  default_mini_params.n_predict) {
                return true;
              }
            }
            // find "User:"
            if (client.output_string.find("User:") != std::string::npos) {
              return true;
            }
          }
          return false;
        };
        if (is_finish()) {
          decoder_helper.send_update(
              {.type = ipcm::DecoderUpdate::Finish, .id = client.req_id});

          // delete only the generated part of the sequence, i.e. keep the
          // system prompt in the cache
          llama_kv_cache_seq_rm(ctx, client.ith_client + 1, -1, -1);
          llama_kv_cache_seq_cp(ctx, 0, client.ith_client + 1, -1, -1);

          const auto t_main_end = ggml_time_us();

          // printf("\033[31mRank %d, Client %3d, seq %3d, prompt %4d t, "
          //        "response %4d t, time %5.2f s, speed %5.2f t/s, cache miss
          //        %d "
          //        "\033[0m\n"
          //        "Input:    %s\n\033[35m"
          //        "Response: %s\033[0m\n\n",
          //        mpi_rank, client.ith_client, client.req_id, client.n_prompt,
          //        client.n_decoded, (t_main_end - client.t_start_prompt) /
          //        1e6, (double)(client.n_prompt + client.n_decoded) /
          //            (t_main_end - client.t_start_prompt) * 1e6,
          //        n_cache_miss, client.input_string.c_str(),
          //        client.output_string.c_str());
          // mark this client as finished
          client.is_running = false;
        }

        client.ith_batch = -1;
      }
    }
  }

  llama_batch_free(batch);
  llama_backend_free();
  MPI_Finalize();
  return 0;
}
