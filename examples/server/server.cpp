#include <algorithm>

#include "common_local.h"
#include "llama-impl.h"
#include "llama.h"
#include "sampling_local.h"
#include "utils.hpp"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

#include <signal.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
};

struct slot_params {
    bool    stream = true;
    int32_t n_keep = 0;  // number of tokens to keep from initial prompt
    int32_t n_discard =
        0;  // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t                       n_predict           = -1;  // new tokens to predict
    bool                          post_sampling_probs = false;
    struct common_params_sampling sampling;
    oaicompat_type                oaicompat             = OAICOMPAT_TYPE_NONE;
    common_chat_format            oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
};

struct server_task {
    int              id    = -1;      // to be filled by server_queue
    int              index = -1;      // used when there are multiple prompts (batch request)
    server_task_type type;
    int              id_target = -1;  // used by SERVER_TASK_TYPE_CANCEL
    slot_params      params;
    llama_tokens     prompt_tokens;
    int              id_selected_slot = -1;

    server_task(server_task_type type) : type(type) {}

    static slot_params params_from_json_cmpl(const common_params & params_base, const json & data) {
        slot_params params;
        slot_params defaults;
        defaults.sampling          = params_base.sampling;
        params.stream              = json_value(data, "stream", false);
        params.post_sampling_probs = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);
        params.n_keep              = json_value(data, "n_keep", defaults.n_keep);

        params.sampling.top_k   = json_value(data, "top_k", defaults.sampling.top_k);
        params.sampling.top_p   = json_value(data, "top_p", defaults.sampling.top_p);
        params.sampling.min_p   = json_value(data, "min_p", defaults.sampling.min_p);
        params.sampling.temp    = json_value(data, "temperature", defaults.sampling.temp);
        params.sampling.n_probs = json_value(data, "n_probs", defaults.sampling.n_probs);

        std::string model_name = params_base.model_alias.empty() ? DEFAULT_OAICOMPAT_MODEL : params_base.model_alias;
        return params;
    }

    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct result_timings {
    int32_t prompt_n = -1;
    double  prompt_ms;
    double  prompt_per_token_ms;
    double  prompt_per_second;

    int32_t predicted_n = -1;
    double  predicted_ms;
    double  predicted_per_token_ms;
    double  predicted_per_second;

    json to_json() const {
        return {
            { "prompt_n",               prompt_n               },
            { "prompt_ms",              prompt_ms              },
            { "prompt_per_token_ms",    prompt_per_token_ms    },
            { "prompt_per_second",      prompt_per_second      },

            { "predicted_n",            predicted_n            },
            { "predicted_ms",           predicted_ms           },
            { "predicted_per_token_ms", predicted_per_token_ms },
            { "predicted_per_second",   predicted_per_second   },
        };
    }
};

struct server_task_result {
    int id      = -1;
    int id_slot = -1;

    virtual int get_index() { return -1; }

    virtual bool is_stop() { return false; }

    virtual json to_json()        = 0;
    virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float       prob;
    std::string text_to_send;

    struct prob_info {
        llama_token tok;
        std::string txt;
        float       prob;
    };

    std::vector<prob_info> probs;

    static float logarithm(float x) {
        // nlohmann::json converts -inf to null, so we need to prevent that
        return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
    }

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
        json out = json::array();
        for (const auto & p : probs) {
            std::string txt(p.text_to_send);
            txt.resize(validate_utf8(txt));
            out.push_back(json{
                { "id",                                               p.tok                                            },
                { "token",                                            txt                                              },
                { "bytes",                                            str_to_bytes(p.text_to_send)                     },
                { post_sampling_probs ? "prob" : "logprob",           post_sampling_probs ? p.prob : logarithm(p.prob) },
                { post_sampling_probs ? "top_probs" : "top_logprobs", p.to_json(post_sampling_probs)                   },
            });
        }
        return out;
    }

    json to_json(bool post_sampling_probs) const {
        json probs_for_token = json::array();
        for (const auto & p : probs) {
            std::string txt(p.txt);
            txt.resize(validate_utf8(txt));
            probs_for_token.push_back(json{
                { "id",                                     p.tok                                            },
                { "token",                                  txt                                              },
                { "bytes",                                  str_to_bytes(p.txt)                              },
                { post_sampling_probs ? "prob" : "logprob", post_sampling_probs ? p.prob : logarithm(p.prob) },
            });
        }
        return probs_for_token;
    }

    static std::vector<unsigned char> str_to_bytes(const std::string & str) {
        std::vector<unsigned char> bytes;
        for (unsigned char c : str) {
            bytes.push_back(c);
        }
        return bytes;
    }
};

struct server_task_result_cmpl_final : server_task_result {
    int index = 0;

    std::string  content;
    llama_tokens tokens;

    bool           stream;
    result_timings timings;
    std::string    prompt;

    bool        truncated;
    int32_t     n_decoded;
    int32_t     n_prompt_tokens;
    int32_t     n_tokens_cached;
    bool        has_new_line;
    std::string stopping_word;
    stop_type   stop = STOP_TYPE_NONE;

    bool                                 post_sampling_probs;
    std::vector<completion_token_output> probs_output;

    slot_params generation_params;

    oaicompat_type     oaicompat             = OAICOMPAT_TYPE_NONE;
    common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    virtual int get_index() override { return index; }

    virtual bool is_stop() override { return true; }

    virtual json to_json() override {
        switch (oaicompat) {
            case OAICOMPAT_TYPE_COMPLETION:
                return to_json_oaicompat();
            case OAICOMPAT_TYPE_CHAT:
                return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
            default:
                GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_oaicompat() {
        std::time_t t        = std::time(0);
        json        logprobs = json(nullptr);  // OAI default to null
        if (!stream && probs_output.size() > 0) {
            logprobs = json{
                { "content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs) },
            };
        }
        json finish_reason = "length";
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = "stop";
        }
        json res = json{
            { "choices",
             json::array({ json{
                  { "text", stream ? "" : content },  // in stream mode, content is already in last partial chunk
                  { "index", index },
                  { "logprobs", logprobs },
                  { "finish_reason", finish_reason },
              } })         },
            { "created", t },
        };
        return res;
    }

    json to_json_oaicompat_chat() {
        std::string     finish_reason = "length";
        common_chat_msg msg;
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            msg           = common_chat_parse(content, oaicompat_chat_format);
            finish_reason = "stop";
        } else {
            msg.content = content;
        }

        json message{
            { "role", "assistant" },
        };
        if (!msg.reasoning_content.empty()) {
            message["reasoning_content"] = msg.reasoning_content;
        }
        if (msg.content.empty()) {
            message["content"] = json();
        } else {
            message["content"] = msg.content;
        }

        json choice{
            { "finish_reason", finish_reason },
            { "index",         0             },
            { "message",       message       },
        };

        if (!stream && probs_output.size() > 0) {
            choice["logprobs"] = json{
                { "content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs) },
            };
        }

        std::time_t t = std::time(0);

        json res = json{
            { "choices", json::array({ choice }) },
            { "created", t                       },
        };

        return res;
    }

    json to_json_oaicompat_chat_stream() {
        std::time_t t             = std::time(0);
        std::string finish_reason = "length";
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = "stop";
        }

        json choice = json{
            { "finish_reason", finish_reason  },
            { "index",         0              },
            { "delta",         json::object() }
        };

        json ret = json{
            { "choices", json::array({ choice }) },
            { "created", t                       },
            { "object",  "chat.completion.chunk" },
            { "usage",
             json{
                  { "completion_tokens", n_decoded },
                  { "prompt_tokens", n_prompt_tokens },
                  { "total_tokens", n_decoded + n_prompt_tokens },
              }                                  },
        };

        return ret;
    }
};

struct server_task_result_cmpl_partial : server_task_result {
    int index = 0;

    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;

    bool                    post_sampling_probs;
    completion_token_output prob_output;
    result_timings          timings;

    // OAI-compat fields
    bool           verbose   = false;
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
    std::string    oaicompat_model;
    std::string    oaicompat_cmpl_id;

    virtual int get_index() override { return index; }

    virtual bool is_stop() override { return false; }

    virtual json to_json() override {
        switch (oaicompat) {
            case OAICOMPAT_TYPE_NONE:
                return to_json_non_oaicompat();
            case OAICOMPAT_TYPE_COMPLETION:
                return to_json_oaicompat();
            case OAICOMPAT_TYPE_CHAT:
                return to_json_oaicompat_chat();
            default:
                GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_non_oaicompat() {
        // non-OAI-compat JSON
        json res = json{
            { "index",            index           },
            { "content",          content         },
            { "tokens",           tokens          },
            { "stop",             false           },
            { "id_slot",          id_slot         },
            { "tokens_predicted", n_decoded       },
            { "tokens_evaluated", n_prompt_tokens },
        };
        // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
        if (timings.prompt_n > 0) {
            res.push_back({ "timings", timings.to_json() });
        }
        if (!prob_output.probs.empty()) {
            res["completion_probabilities"] =
                completion_token_output::probs_vector_to_json({ prob_output }, post_sampling_probs);
        }
        return res;
    }

    json to_json_oaicompat() {
        std::time_t t        = std::time(0);
        json        logprobs = json(nullptr);  // OAI default to null
        if (prob_output.probs.size() > 0) {
            logprobs = json{
                { "content", completion_token_output::probs_vector_to_json({ prob_output }, post_sampling_probs) },
            };
        }
        json res = json{
            { "choices", json::array({ json{
                             { "text", content },
                             { "index", index },
                             { "logprobs", logprobs },
                             { "finish_reason", nullptr },
                         } }) },
            { "created", t                            },
        };

        if (timings.prompt_n >= 0) {
            res.push_back({ "timings", timings.to_json() });
        }

        return res;
    }

    json to_json_oaicompat_chat() {
        bool        first = n_decoded == 0;
        std::time_t t     = std::time(0);
        json        choices;

        if (first) {
            if (content.empty()) {
                choices = json::array({
                    json{
                         { "finish_reason", nullptr }, { "index", 0 }, { "delta", json{ { "role", "assistant" } } } }
                });
            } else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{
                    { "choices", json::array({ json{ { "finish_reason", nullptr },
                                                     { "index", 0 },
                                                     { "delta", json{ { "role", "assistant" } } } } }) },
                    { "created", t                                                                                                     }
                };

                json second_ret = json{
                    { "choices", json::array({ json{ { "finish_reason", nullptr },
                                                     { "index", 0 },
                                                     { "delta", json{ { "content", content } } } } }) },
                    { "created", t                                                                                                    }
                };

                return std::vector<json>({ initial_ret, second_ret });
            }
        } else {
            choices = json::array({
                json{
                     { "finish_reason", nullptr },
                     { "index", 0 },
                     { "delta",
                      json{
                          { "content", content },
                      } },
                     }
            });
        }

        GGML_ASSERT(choices.size() >= 1);

        if (prob_output.probs.size() > 0) {
            choices[0]["logprobs"] = json{
                { "content", completion_token_output::probs_vector_to_json({ prob_output }, post_sampling_probs) },
            };
        }

        json ret = json{
            { "choices", choices },
            { "created", t       }
        };

        if (timings.prompt_n >= 0) {
            ret.push_back({ "timings", timings.to_json() });
        }

        return std::vector<json>({ ret });
    }
};

struct server_slot {
    int id;
    int id_task = -1;

    server_task_type task_type = SERVER_TASK_TYPE_COMPLETION;

    llama_batch batch_spec = {};

    llama_context * ctx     = nullptr;
    llama_context * ctx_dft = nullptr;

    // the index relative to completion multi-task request
    size_t index = 0;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1;  // TODO: disambiguate from params.n_predict

    // n_prompt_tokens may not be equal to prompt_tokens.size(), because prompt maybe truncated
    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    // input prompt tokens
    llama_tokens prompt_tokens;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    llama_tokens generated_tokens;

    llama_tokens cache_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool      has_next_token = true;
    bool      has_new_line   = false;
    bool      truncated      = false;
    stop_type stop;

    std::string stopping_word;

    struct common_sampler * smpl = nullptr;

    llama_token sampled;

    // stats
    size_t n_sent_text = 0;  // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing;  // ms
    double t_token_generation;   // ms

    std::function<void(int)> callback_on_release;

    void reset() {
        n_prompt_tokens = 0;
        last_nl_pos     = 0;
        generated_text  = "";
        has_new_line    = false;
        truncated       = false;
        stop            = STOP_TYPE_NONE;
        stopping_word   = "";
        n_past          = 0;
        n_sent_text     = 0;
        task_type       = SERVER_TASK_TYPE_COMPLETION;

        generated_tokens.clear();
        generated_token_probs.clear();
    }

    bool has_budget(const common_params & global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true;  // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0;  // no budget
    }

    bool is_processing() const { return state != SLOT_STATE_IDLE; }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release() {
        if (is_processing()) {
            t_last_used        = ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
            state              = SLOT_STATE_IDLE;
            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        return timings;
    }

    json to_json() const {
        return json{
            { "id", id },
            { "id_task", id_task },
            { "n_ctx", n_ctx },
            { "is_processing", is_processing() },
            { "prompt", common_detokenize(ctx, prompt_tokens) },
            { "next_token",
             {
                  { "has_next_token", has_next_token },
                  { "has_new_line", has_new_line },
                  { "n_remain", n_remaining },
                  { "n_decoded", n_decoded },
                  { "stopping_word", stopping_word },
              } },
        };
    }
};

struct server_queue {
    int  id = 0;
    bool running;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex              mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task)> callback_new_task;
    std::function<void(void)>        callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        GGML_ASSERT(task.id != -1);
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
        condition_tasks.notify_one();
        return task.id;
    }

    // multi-task version of post()
    int post(std::vector<server_task> & tasks, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : tasks) {
            if (task.id == -1) {
                task.id = id++;
            }
            // if this is cancel task make sure to clean up pending tasks
            if (task.type == SERVER_TASK_TYPE_CANCEL) {
                cleanup_pending_task(task.id_target);
            }
            if (front) {
                queue_tasks.push_front(std::move(task));
            } else {
                queue_tasks.push_back(std::move(task));
            }
        }
        condition_tasks.notify_one();
        return 0;
    }

    // Add a new task, but defer until one slot is available
    void defer(server_task task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        queue_tasks_deferred.push_back(std::move(task));
        condition_tasks.notify_one();
    }

    // Get the next id for creating a new task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int                          new_id = id++;
        return new_id;
    }

    // Register function to process a new task
    void on_new_task(std::function<void(server_task)> callback) { callback_new_task = std::move(callback); }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) { callback_update_slots = std::move(callback); }

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    void pop_deferred_task() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!queue_tasks_deferred.empty()) {
            queue_tasks.emplace_back(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
        condition_tasks.notify_one();
    }

    // end the start_loop routine
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop() {
        running = true;

        while (true) {
            while (true) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (!running) {
                    return;
                }
                if (queue_tasks.empty()) {
                    lock.unlock();
                    break;
                }
                server_task task = queue_tasks.front();
                queue_tasks.pop_front();
                lock.unlock();

                callback_new_task(std::move(task));
            }

            // all tasks in the current loop is processed, slots data is now ready

            callback_update_slots();

            {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (!running) {
                    return;
                }
                if (queue_tasks.empty()) {
                    condition_tasks.wait(lock, [&] { return (!queue_tasks.empty() || !running); });
                }
            }
        }
    }

  private:
    void cleanup_pending_task(int id_target) {
        // no need lock because this is called exclusively by post()
        auto rm_func = [id_target](const server_task & task) {
            return task.id_target == id_target;
        };
        queue_tasks.erase(std::remove_if(queue_tasks.begin(), queue_tasks.end(), rm_func), queue_tasks.end());
        queue_tasks_deferred.erase(std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
                                   queue_tasks_deferred.end());
    }
};

struct server_response {
    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex              mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task) {
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(id_task);
    }

    void add_waiting_tasks(const std::vector<server_task> & tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & task : tasks) {
            waiting_task_ids.insert(task.id);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(id_task);
        // make sure to clean up all pending results
        queue_results.erase(
            std::remove_if(queue_results.begin(), queue_results.end(),
                           [id_task](const server_task_result_ptr & res) { return res->id == id_task; }),
            queue_results.end());
    }

    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & id_task : id_tasks) {
            waiting_task_ids.erase(id_task);
        }
    }

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&] { return !queue_results.empty(); });

            for (size_t i = 0; i < queue_results.size(); i++) {
                if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                    server_task_result_ptr res = std::move(queue_results[i]);
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);

            for (int i = 0; i < (int) queue_results.size(); i++) {
                if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                    server_task_result_ptr res = std::move(queue_results[i]);
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }

            std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
            if (cr_res == std::cv_status::timeout) {
                return nullptr;
            }
        }

        // should never reach here
    }

    // single-task version of recv()
    server_task_result_ptr recv(int id_task) {
        std::unordered_set<int> id_tasks = { id_task };
        return recv(id_tasks);
    }

    std::vector<server_task_result_ptr> send_buffer;

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result) { send_buffer.emplace_back(std::move(result)); }

    void send_commit() {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (auto & result : send_buffer) {
            if (std::find_if(waiting_task_ids.begin(), waiting_task_ids.end(),
                             [&result](int id) { return id == result->id; }) != waiting_task_ids.end()) {
                queue_results.emplace_back(std::move(result));
            }
        }
        condition_results.notify_all();
        send_buffer.clear();
    }
};

struct server_context {
    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result llama_init;
    common_init_result llama_init_dft;

    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    const llama_vocab * vocab = nullptr;

    llama_model * model_dft = nullptr;

    llama_context_params cparams_dft;

    llama_batch batch = {};

    bool clean_kv_cache = true;
    bool add_bos_token  = true;

    int32_t n_ctx;  // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    server_queue    queue_tasks;
    server_response queue_results;

    common_chat_templates_ptr chat_templates;

    ~server_context() {
        // Clear any sampling context
        for (server_slot & slot : slots) {
            common_sampler_free(slot.smpl);
            slot.smpl = nullptr;

            llama_free(slot.ctx_dft);
            slot.ctx_dft = nullptr;

            llama_batch_free(slot.batch_spec);
        }

        llama_batch_free(batch);
    }

    bool load_model(const common_params & params) {
        params_base = params;
        llama_init  = common_init_from_params(params_base);
        model       = llama_init.model.get();
        ctx         = llama_init.context.get();
        if (model == nullptr) {
            return false;
        }
        vocab          = llama_model_get_vocab(model);
        n_ctx          = llama_n_ctx(ctx);
        add_bos_token  = llama_vocab_get_add_bos(vocab);
        chat_templates = common_chat_templates_init(model, params_base.chat_template);
        common_chat_format_example(chat_templates.get());
        return true;
    }

    void init() {
        const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;
        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;
            slot.id                  = i;
            slot.ctx                 = ctx;
            slot.n_ctx               = n_ctx_slot;
            slot.n_predict           = params_base.n_predict;
            slot.params.sampling     = params_base.sampling;
            slot.callback_on_release = [this](int) {
                queue_tasks.pop_deferred_task();
            };
            slot.reset();
            slots.push_back(slot);
        }
        {
            const int32_t n_batch = llama_n_batch(ctx);
            // only a single seq_id per token is needed
            batch                 = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }
    }

    server_slot * get_slot_by_id(int id) {
        for (server_slot & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;
        // int lcs_len = 0;
        // for (server_slot & slot : slots) {
        //     // skip the slot if it is not available
        //     if (slot.is_processing()) {
        //         continue;
        //     }
        //     // skip the slot if it does not contains cached tokens
        //     if (slot.cache_tokens.empty()) {
        //         continue;
        //     }
        //     ret = &slot; break;
        //   //// length of the Longest Common Subsequence between the current slot's prompt and the input prompt
        //   //int cur_lcs_len = common_lcs(slot.cache_tokens, task.prompt_tokens);
        //   //// select the current slot if the criteria match
        //   //if (cur_lcs_len > lcs_len) {
        //   //    lcs_len = cur_lcs_len;
        //   //    ret = &slot;
        //   //}
        // }
        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = ggml_time_us();
            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }
                // select the current slot if the criteria match
                if (slot.t_last_used < t_last) {
                    t_last = slot.t_last_used;
                    ret    = &slot;
                }
            }
        }

        return ret;
    }

    bool launch_slot_with_task(server_slot & slot, const server_task & task) {
        slot.reset();
        slot.id_task       = task.id;
        slot.index         = task.index;
        slot.task_type     = task.type;
        slot.params        = std::move(task.params);
        slot.prompt_tokens = std::move(task.prompt_tokens);
        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            slot.params.n_predict = slot.n_predict;
        }
        {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }

            slot.smpl = common_sampler_init(model, slot.params.sampling);
            if (slot.smpl == nullptr) {
                return false;
            }
        }
        if (slot.ctx_dft) {
            llama_batch_free(slot.batch_spec);
            slot.batch_spec = llama_batch_init(1, 0, 1);
        }
        slot.state = SLOT_STATE_STARTED;
        return true;
    }

    void kv_cache_clear() {
        // clear the entire KV cache
        llama_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled                = result.tok;
        slot.generated_text += token_str;
        slot.generated_tokens.push_back(result.tok);
        slot.has_next_token = true;
        // check if there is incomplete UTF-8 character at the end
        bool incomplete     = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t            pos       = std::min(slot.n_sent_text, slot.generated_text.size());
            const std::string str_test  = slot.generated_text.substr(pos);
            bool              send_text = true;
            size_t            stop_pos  = std::string::npos;
            if (slot.has_next_token) {
                stop_pos  = std::string::npos;
                send_text = stop_pos == std::string::npos;
            }
            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.params.stream) {
                send_partial_response(slot, result);
            }
        }
        if (incomplete) {
            slot.has_next_token = true;
        }
        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;
        }
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;
        }
        if (slot.n_past + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;
        }
        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;
        }
        const auto n_ctx_train = llama_model_n_ctx_train(model);
        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;  // stop prediction
        }
        return slot.has_next_token;       // continue
    }

    void send_final_response(server_slot & slot) {
        auto res                   = std::make_unique<server_task_result_cmpl_final>();
        res->id                    = slot.id_task;
        res->id_slot               = slot.id;
        res->index                 = slot.index;
        res->content               = std::move(slot.generated_text);
        res->tokens                = std::move(slot.generated_tokens);
        res->timings               = slot.get_timings();
        res->prompt                = common_detokenize(ctx, slot.prompt_tokens, true);
        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.n_prompt_tokens;
        res->n_tokens_cached       = slot.n_past;
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.params.post_sampling_probs;
        res->stream                = slot.params.stream;
        res->oaicompat             = slot.params.oaicompat;
        res->oaicompat_chat_format = slot.params.oaicompat_chat_format;

        // populate res.probs_output
        if (slot.params.sampling.n_probs > 0) {
            if (!slot.params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output  = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(), slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin(),
                                                                         slot.generated_token_probs.end());
            }
        }
        res->generation_params = slot.params;  // copy the parameters
        queue_results.send(std::move(res));
        //llama_perf_context_print(ctx);
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id      = slot.id_task;
        res->index   = slot.index;
        res->content = tkn.text_to_send;
        res->tokens  = { tkn.tok };

        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.n_prompt_tokens;
        res->post_sampling_probs = slot.params.post_sampling_probs;
        res->oaicompat           = slot.params.oaicompat;

        // populate res.probs_output
        if (slot.params.sampling.n_probs > 0) {
            res->prob_output = tkn;  // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void cancel_tasks(const std::unordered_set<int> & id_tasks) {
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(task);
        }
        queue_tasks.post(cancel_tasks, true);
    }

    void receive_multi_results(const std::unordered_set<int> &                                    id_tasks,
                               const std::function<void(std::vector<server_task_result_ptr> &)> & result_handler,
                               const std::function<bool()> & is_connection_closed) {
        std::vector<server_task_result_ptr> results(id_tasks.size());
        for (int i = 0; i < (int) id_tasks.size(); i++) {
            server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);
            if (is_connection_closed()) {
                cancel_tasks(id_tasks);
                return;
            }
            if (result == nullptr) {
                i--;
                continue;
            }
            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final *>(result.get()) != nullptr);
            const size_t idx = result->get_index();
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = std::move(result);
        }
        result_handler(results);
    }

    // receive the results from task(s), in stream mode
    void receive_cmpl_results_stream(const std::unordered_set<int> &                       id_tasks,
                                     const std::function<bool(server_task_result_ptr &)> & result_handler,
                                     const std::function<bool()> &                         is_connection_closed) {
        size_t n_finished = 0;
        while (true) {
            server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);

            if (is_connection_closed()) {
                cancel_tasks(id_tasks);
                return;
            }

            if (result == nullptr) {
                continue;  // retry
            }

            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_cmpl_final *>(result.get()) != nullptr);
            if (!result_handler(result)) {
                cancel_tasks(id_tasks);
                break;
            }

            if (result->is_stop()) {
                if (++n_finished == id_tasks.size()) {
                    break;
                }
            }
        }
    }

    void process_single_task(server_task task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
                {
                    const int     id_slot = task.id_selected_slot;
                    server_slot * slot    = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);
                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        queue_tasks.defer(task);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        queue_tasks.defer(task);
                        break;
                    }

                    if (!launch_slot_with_task(*slot, task)) {
                        break;
                    }
                }
                break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    for (auto & slot : slots) {
                        if (slot.id_task == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                }
                break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                }
                break;
        }
    }

    void update_slots() {
        bool all_idle = true;
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                all_idle = false;
                break;
            }
        }
        if (all_idle) {
            if (clean_kv_cache) {
                kv_cache_clear();
            }
            return;
        }
        server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
        task.id = queue_tasks.get_new_id();
        queue_tasks.post(task);

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.n_past + 1 >= slot.n_ctx) {
                const int n_keep    = slot.params.n_keep + add_bos_token;
                const int n_left    = slot.n_past - n_keep;
                const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);
                llama_kv_cache_seq_rm(ctx, slot.id, n_keep, n_keep + n_discard);
                llama_kv_cache_seq_add(ctx, slot.id, n_keep + n_discard, slot.n_past, -n_discard);
                for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++) {
                    slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                }
                slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
                slot.n_past -= n_discard;
                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING) {
                // check if we can batch this slot with the previous one
                if (!slot_batched) {
                    slot_batched = &slot;
                }
                slot.i_batch = batch.n_tokens;
                common_batch_add(batch, slot.sampled, slot.n_past, { slot.id }, true);
                slot.n_past += 1;
                slot.cache_tokens.push_back(slot.sampled);
            }
        }

        // process in chunks of params.n_batch
        int32_t n_batch = llama_n_batch(ctx);

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // check if we can batch this slot with the previous one
                if (slot.is_processing()) {
                    if (!slot_batched) {
                        slot_batched = &slot;
                    }
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    auto & prompt_tokens = slot.prompt_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation     = 0;

                        slot.n_past          = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();
                        slot.state           = SLOT_STATE_PROCESSING_PROMPT;

                        // empty prompt passed -> release the slot and send empty response
                        if (prompt_tokens.empty()) {
                            slot.release();
                            send_final_response(slot);
                            continue;
                        }
                        if (slot.params.n_keep < 0) {
                            slot.params.n_keep = slot.n_prompt_tokens;
                        }
                        slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                        // if input prompt is too big, truncate it
                        if (slot.n_prompt_tokens >= slot.n_ctx) {
                            const int n_left = slot.n_ctx - slot.params.n_keep;

                            const int n_block_size = n_left / 2;
                            const int erased_blocks =
                                (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                            llama_tokens new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + slot.params.n_keep);

                            new_tokens.insert(new_tokens.end(),
                                              prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                              prompt_tokens.end());

                            prompt_tokens = std::move(new_tokens);

                            slot.truncated       = true;
                            slot.n_prompt_tokens = prompt_tokens.size();

                            GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                        }
                        // reuse any previously computed tokens that are common with the new prompt
                        slot.n_past = common_lcp(slot.cache_tokens, prompt_tokens);

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            slot.n_past--;
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    // keep only the common part
                    if (!llama_kv_cache_seq_rm(ctx, slot.id, slot.n_past, -1)) {
                        // could not partially delete (likely using a non-Transformer model)

                        // there is no common part left
                        slot.n_past = 0;
                    }

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);

                    // add prompt tokens for processing in the current batch
                    while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                        const bool need_embd = llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE;
                        common_batch_add(batch, prompt_tokens[slot.n_past], slot.n_past, { slot.id }, need_embd);
                        slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        slot.n_prompt_tokens_processed++;
                        slot.n_past++;
                    }

                    // entire prompt has been processed
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        common_sampler_reset(slot.smpl);

                        // Process all prompt tokens through sampler system
                        for (int i = 0; i < slot.n_prompt_tokens; ++i) {
                            common_sampler_accept(slot.smpl, prompt_tokens[i], false);
                        }

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;
                    }
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        if (slot_batched) {
            // make sure we're in the right embedding mode
            llama_set_embeddings(ctx, false);
        }
        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,           batch.token + i,  nullptr,          batch.pos + i,
                batch.n_seq_id + i, batch.seq_id + i, batch.logits + i,
            };

            const int ret = llama_decode(ctx, batch_view);

            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    for (auto & slot : slots) {
                        slot.release();
                    }
                    break;  // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;  // continue loop of n_batch
            }

            if (params_base.display_chat) {
                for (auto & slot : slots) {
                    std::string str       = common_detokenize(ctx, slot.cache_tokens);
                    int         start_idx = str.find_last_of('\n') + 1;
                    if (start_idx == std::string::npos) {
                        start_idx = 0;
                    }
                    str = str.substr(start_idx);
                    printf("slot[%d] : %s\n", slot.id, str.c_str());
                }
                std::cout << "\033[2J\033[1;1H";
            }

            for (auto & slot : slots) {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue;  // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue;  // continue loop of slots
                }

                const int tok_idx = slot.i_batch - i;
                int       id      = common_sampler_sample(slot.smpl, this->ctx, tok_idx);
                slot.i_batch      = -1;
                common_sampler_accept(slot.smpl, id, true);
                slot.n_decoded += 1;
                const int64_t t_current = ggml_time_us();
                if (slot.n_decoded == 1) {
                    slot.t_start_generation  = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                }
                slot.t_token_generation = (t_current - slot.t_start_generation) / 1e3;

                auto accept_special_token = [&](server_slot & slot, llama_token token) {
                    return params_base.special || slot.params.sampling.preserved_tokens.find(token) !=
                                                      slot.params.sampling.preserved_tokens.end();
                };

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f;  // TODO: set it here instead of doing inside populate_token_probs
                GGML_ASSERT(slot.params.sampling.n_probs <= 0);
                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.release();
                    send_final_response(slot);
                }
            }
        }
        queue_results.send_commit();
    }
};

int main(int argc, char ** argv) {
    // own arguments required by this example
    common_params params;

    // Default configuration file path
    std::string config_file = "config.yaml";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                fprintf(stderr, "Error: --config requires a file path\n");
                return 1;
            }
        }
    }

    // Try to load parameters from YAML configuration file
    if (std::filesystem::exists(config_file)) {
        printf("Loading parameters from %s\n", config_file.c_str());
        if (!params.load_from_yaml(config_file)) {
            fprintf(stderr, "Failed to load parameters from %s, using defaults\n", config_file.c_str());
        }
    } else {
        printf("Configuration file %s not found, using default parameters\n", config_file.c_str());
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    //llama_numa_init(params.numa);

    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());

    auto res_ok = [](httplib::Response & res, const json & data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    };

    const auto handle_completions_impl = [&ctx_server, &res_ok](server_task_type type, json & data,
                                                                const std::function<bool()> & is_connection_closed,
                                                                httplib::Response & res, oaicompat_type oaicompat) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION);

        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(type);

            task.id    = ctx_server.queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens    = std::move(tokenized_prompts[i]);
            task.params           = server_task::params_from_json_cmpl(ctx_server.params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat = oaicompat;

            tasks.push_back(task);
        }

        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool       stream   = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            ctx_server.receive_multi_results(
                task_ids,
                [&](std::vector<server_task_result_ptr> & results) {
                    if (results.size() == 1) {
                        // single result
                        res_ok(res, results[0]->to_json());
                    } else {
                        // multiple results (multitask)
                        json arr = json::array();
                        for (auto & res : results) {
                            arr.push_back(res->to_json());
                        }
                        res_ok(res, arr);
                    }
                },
                is_connection_closed);

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            const auto chunked_content_provider = [task_ids, &ctx_server, oaicompat](size_t, httplib::DataSink & sink) {
                ctx_server.receive_cmpl_results_stream(
                    task_ids,
                    [&](server_task_result_ptr & result) -> bool {
                        json res_json = result->to_json();
                        if (res_json.is_array()) {
                            for (const auto & res : res_json) {
                                if (!server_sent_event(sink, "data", res)) {
                                    // sending failed (HTTP connection closed), cancel the generation
                                    return false;
                                }
                            }
                            return true;
                        } else {
                            return server_sent_event(sink, "data", res_json);
                        }
                    },
                    [&sink]() {
                        // note: do not use req.is_connection_closed here because req is already destroyed
                        return !sink.is_writable();
                    });
                if (oaicompat != OAICOMPAT_TYPE_NONE) {
                    static const std::string ev_done = "data: [DONE]\n\n";
                    sink.write(ev_done.data(), ev_done.size());
                }
                sink.done();
                return false;
            };

            auto on_complete = [task_ids, &ctx_server](bool /*unused*/) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_completions_oai = [&handle_completions_impl](const httplib::Request & req,
                                                                   httplib::Response &      res) {
        json data = oaicompat_completion_params_parse(json::parse(req.body));
        return handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, req.is_connection_closed, res,
                                       OAICOMPAT_TYPE_COMPLETION);
    };

    const auto handle_chat_completions = [&ctx_server, &handle_completions_impl](const httplib::Request & req,
                                                                                 httplib::Response &      res) {
        auto body = json::parse(req.body);
        json data = oaicompat_completion_params_parse(body, ctx_server.chat_templates.get());

        return handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, req.is_connection_closed, res,
                                       OAICOMPAT_TYPE_CHAT);
    };

    // register API routes
    svr->Post("/v1/completions", handle_completions_oai);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Get("/exit", [](const httplib::Request & /*unused*/, httplib::Response & /*unused*/) { exit(0); });

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    svr->new_task_queue = [&params] {
        return new httplib::ThreadPool(params.n_threads_http);
    };

    // clean up function, to be called before exit
    auto clean_up = [&svr]() {
        svr->stop();
        llama_backend_free();
    };

    svr->bind_to_port(params.hostname, params.port);

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    printf("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__,
           params.hostname.c_str(), params.port, params.n_threads_http);

    // load the model

    if (!ctx_server.load_model(params)) {
        clean_up();
        // t.join(); // FIXME: see below
        return 1;
    }

    ctx_server.init();

    ctx_server.queue_tasks.on_new_task(
        [&ctx_server](const server_task & task) { ctx_server.process_single_task(task); });

    ctx_server.queue_tasks.on_update_slots([&ctx_server]() { ctx_server.update_slots(); });

    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.queue_tasks.start_loop();

    clean_up();

    return 0;
}
