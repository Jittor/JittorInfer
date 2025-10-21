// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.
#include "common_local.h"
#include "ggml.h"
#include "llama.h"
#include "sampling_local.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string &str) {
  size_t start = 0;
  size_t end = str.size();

  while (start < end && isspace(str[start])) {
    start += 1;
  }

  while (end > start && isspace(str[end - 1])) {
    end -= 1;
  }

  return str.substr(start, end - start);
}

static std::string k_system =
    R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:)";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you "
    "explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
};

struct client {
  ~client() {
    if (smpl) {
      common_sampler_local_free(smpl);
    }
  }

  int32_t id = 0;
  llama_seq_id seq_id = -1;
  llama_token sampled;

  int64_t t_start_prompt;
  int64_t t_start_gen;

  int32_t n_prompt = 0;
  int32_t n_decoded = 0;
  int32_t i_batch = -1;

  std::string input;
  std::vector<llama_token> prompt;
  std::string response;

  struct common_sampler_local *smpl = nullptr;

  void reset(std::string _input, std::vector<llama_token> _prompt) {
    static llama_seq_id g_seq_id = 0;
    seq_id = g_seq_id++;

    t_start_prompt = ggml_time_us();
    t_start_gen = 0;

    input = std::move(_input);
    prompt = std::move(_prompt);
    response = "";
    n_prompt = prompt.size();
    n_decoded = 0;

    common_sampler_reset(smpl);
  }
};

static void print_date_time() {
  std::time_t current_time = std::time(nullptr);
  std::tm *local_time = std::localtime(&current_time);
  char buffer[80];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

  printf("\n");
  printf("\033[35mrun parameters as of %s\033[0m\n", buffer);
  printf("\n");
}

static struct DefaultMiniParams {
  int main_gpu = 1;
  int n_gpu_layers = 99;
  enum llama_split_mode split_mode =
      LLAMA_SPLIT_MODE_ROW; // how to split the model across GPUs
  float tensor_split[128] = {
      0};                 // how split tensors should be distributed across GPUs
  bool use_mmap = true;   // use mmap for faster loads
  bool use_mlock = false; // use mlock to keep model in memory
  bool check_tensors = false; // validate tensor data

  std::string model = "/root/data/DeepSeek-V2-Lite-Chat-f16.gguf";

  uint32_t n_ctx = 512;    // context size
  uint32_t n_threads = 64; // number of threads to use for computation
  uint32_t n_threads_batch =
      64; // number of threads to use for batch processing

  float defrag_thold = 0.1f; // defragmentation threshold
  bool no_perf = false;      // disable performance metrics
  std::vector<common_adapter_lora_info>
      lora_adapters; // lora adapter path with user defined scale

  int32_t n_batch = 512; // logical batch size for prompt processing
  bool enable_chat_template = true;
  bool escape = true;
  common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;

  // parallel test configs
  int n_parallel = 1;
  bool cont_batching = true;
  bool dump_kv_cache = false;
  int32_t n_predict = -1; // new tokens to predict

} default_mini_params;

static llama_model_params common_model_params_to_llama_local() {
  auto mparams = llama_model_default_params();

  mparams.main_gpu = default_mini_params.main_gpu;
  mparams.split_mode = default_mini_params.split_mode;
  mparams.tensor_split = default_mini_params.tensor_split;
  mparams.use_mmap = default_mini_params.use_mmap;
  mparams.use_mlock = default_mini_params.use_mlock;
  mparams.check_tensors = default_mini_params.check_tensors;
  mparams.n_gpu_layers = default_mini_params.n_gpu_layers;
  // 开启张量并行
  mparams.enable_tensor_parallel = true;
  mparams.enable_expert_parallel = true;
  mparams.enable_data_parallel = true;
  mparams.enable_fused_moe = true;
  mparams.enable_mpi = true;

#ifdef LLAMA_MPI_SUPPORT
  if (mparams.enable_mpi) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mparams.tp_id = rank;
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    mparams.num_parallel = num_processes;
  }
#endif

  mparams.kv_overrides = NULL;
  return mparams;
}

static llama_context_params common_context_params_to_llama_local() {
  auto cparams = llama_context_default_params();

  cparams.n_ctx = default_mini_params.n_ctx;
  cparams.n_threads = default_mini_params.n_threads;
  cparams.n_threads_batch = default_mini_params.n_threads_batch;
  cparams.defrag_thold = default_mini_params.defrag_thold;
  cparams.no_perf = default_mini_params.no_perf;

  return cparams;
}

static common_init_result common_init_from_params_local() {
  common_init_result iparams;

  auto mparams = common_model_params_to_llama_local();

  llama_model *model = nullptr;

  model =
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

  std::cout << __func__
            << ": warming up the model with an empty run - please wait ... "
               "(--no-warmup to disable)\n";

  std::vector<llama_token> tmp;

  llama_token bos = llama_vocab_bos(vocab);
  llama_token eos = llama_vocab_eos(vocab);

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
// 得到所有server下一步需要处理的token数量
static void recv_batch_list(int *all_server_tokens, int *self_server_tokens) {
  MPI_Allgather(self_server_tokens, 1, MPI_INT, all_server_tokens, 1, MPI_INT,
                MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  srand(1234);

  int mpi_rank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // number of simultaneous "clients" to simulate
  const int32_t n_clients = default_mini_params.n_parallel;

  // dedicate one sequence to the system prompt
  default_mini_params.n_parallel += 1;

  // insert new requests as soon as the previous one is done
  const bool cont_batching = default_mini_params.cont_batching;

  const bool dump_kv_cache = default_mini_params.dump_kv_cache;

  // init llama.cpp
  llama_backend_init();
  // llama_numa_init(params.numa);

  // load the target model
  common_init_result llama_init = common_init_from_params_local();

  llama_model *model = llama_init.model.get();
  llama_context *ctx = llama_init.context.get();

  const llama_vocab *vocab = llama_model_get_vocab(model);

  const int n_ctx = llama_n_ctx(ctx);

  common_params_local_sampling sparams;
  std::vector<client> clients(n_clients);
  for (size_t i = 0; i < clients.size(); ++i) {
    auto &client = clients[i];
    client.id = i;
    client.smpl = common_sampler_local_init(model, sparams);
  }

  std::vector<llama_token> tokens_system;
  tokens_system = common_tokenize(ctx, k_system, true);
  const int32_t n_tokens_system = tokens_system.size();

  // the max batch size is as large as the context to handle cases where we get
  // very long input prompt from multiple users. regardless of the size, the
  // main loop will chunk the batch into a maximum of params.n_batch tokens at a
  // time
  llama_batch batch = llama_batch_init(n_ctx, 0, 1);

  // int32_t n_total_prompt = 0;
  // int32_t n_total_gen    = 0;
  int32_t n_cache_miss = 0;

  struct llama_kv_cache_view kvc_view =
      llama_kv_cache_view_init(ctx, n_clients);

  const auto t_main_start = ggml_time_us();

  printf("%s: n_parallel = %d, cont_batching = %d, system "
         "tokens = %d\n",
         __func__, n_clients, cont_batching, n_tokens_system);

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
      if (client.seq_id == -1) {
        continue;
      }
      client.i_batch = batch.n_tokens;
      common_batch_add(batch, client.sampled,
                       n_tokens_system + client.n_prompt + client.n_decoded,
                       {client.id + 1}, true);
      client.n_decoded += 1;
    }

    if (batch.n_tokens == 0) {
      // all sequences have ended - clear the entire KV cache
      for (int i = 1; i <= n_clients; ++i) {
        llama_kv_cache_seq_rm(ctx, i, -1, -1);
        // but keep the system prompt
        llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
      }
      printf("%s: clearing the KV cache\n", __func__);
    }

    // insert new sequences for decoding
    // 这个阶段是prefill，结束后将所有client的tokens连成一个batch，唯一一个n_tokens很长的阶段，decode阶段g_seq_id
    // == n_seq
    if (cont_batching || batch.n_tokens == 0) {
      for (auto &client : clients) {
        if (client.seq_id == -1) {
          {
            std::string input = k_prompts[rand() % k_prompts.size()];
            std::vector<llama_token> prompt =
                common_tokenize(ctx, input + "\nAssistant:", false);
            client.reset(std::move(input), std::move(prompt));
          }

          for (size_t i = 0; i < client.prompt.size(); ++i) {
            common_batch_add(batch, client.prompt[i], i + n_tokens_system,
                             {client.id + 1}, false);
          }

          // extract the logits only for the last token
          if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
          }

          client.i_batch = batch.n_tokens - 1;

          printf("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n",
                 client.id, client.seq_id);
        }
      }
    }

    if (batch.n_tokens == 0) {
      break;
    }

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
        if (client.i_batch < (int)i || client.i_batch >= (int)(i + n_tokens)) {
          continue;
        }

        const llama_token id =
            common_sampler_sample(client.smpl, ctx, client.i_batch - i);

        common_sampler_accept(client.smpl, id, true);

        if (client.n_decoded == 1) {
          client.t_start_gen = ggml_time_us();
        }

        const std::string token_str = common_token_to_piece(ctx, id);

        client.response += token_str;
        client.sampled = id;

        if (client.n_decoded > 2 &&
            (llama_vocab_is_eog(vocab, id) ||
             (default_mini_params.n_predict > 0 &&
              client.n_decoded + client.n_prompt >=
                  default_mini_params.n_predict) ||
             client.response.find("User:") != std::string::npos ||
             client.response.find('\n') != std::string::npos)) {
          // basic reverse prompt
          const size_t pos = client.response.find("User:");
          if (pos != std::string::npos) {
            client.response = client.response.substr(0, pos);
          }

          // delete only the generated part of the sequence, i.e. keep the
          // system prompt in the cache
          llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
          llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

          const auto t_main_end = ggml_time_us();

          printf("\033[31mRank %d, Client %3d, seq %3d, prompt %4d t, "
                 "response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d "
                 "\033[0m"
                 "Input:    %s\n\033[35m"
                 "Response: %s\033[0m\n\n",
                 mpi_rank, client.id, client.seq_id, client.n_prompt,
                 client.n_decoded, (t_main_end - client.t_start_prompt) / 1e6,
                 (double)(client.n_prompt + client.n_decoded) /
                     (t_main_end - client.t_start_prompt) * 1e6,
                 n_cache_miss, ::trim(client.input).c_str(),
                 ::trim(client.response).c_str());
          // mark this client as finished
          client.seq_id = -1;
        }

        client.i_batch = -1;
      }
    }
    // printf("input %s; output %s\n\n", ::trim(clients[0].input).c_str(),
    // ::trim(clients[0].response).c_str()); printf("input %s; output %s\n\n",
    // ::trim(clients[1].input).c_str(), ::trim(clients[1].response).c_str());
  }

  while (llama_empty_run(ctx)) {
    ;
  }

  const auto t_main_end = ggml_time_us();

  printf("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",
         (double)(llama_all_processed_tokens(ctx)) /
             (t_main_end - t_main_start) * 1e6);

  llama_batch_free(batch);

  llama_backend_free();

  // printf("\n\n");

  MPI_Finalize();

  return 0;
}