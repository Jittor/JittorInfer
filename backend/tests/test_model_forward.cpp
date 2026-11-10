/// @file test_model_forward.cpp
/// @brief 快速调试入口：不起前端、不连 ZMQ，直接驱动 ModelRunner 跑
///        「prefill + 贪心/按 seed 采样 decode」。
///
/// 为什么需要它：接新计算图（视觉编码、chunk-GDR 动态 prefill）时走
/// engine_server 要拉前端、握手、发 HTTP，反馈太慢；这里加载 runner 后直接喂
/// token、打印图清单 / top-k /
/// 每步耗时，几十秒就能看出"新图能不能跑、数值对不对"。
///
/// **它不是调度器**：图选择与每槽分配只做"最小可用"版本（逐图评估容量 +
/// 每槽上限 clamp），真正的调度策略在 server/scheduler.hpp（分块
/// prefill、混合批、动态图换取 少一步 prefill
/// 等都不在这里）。所以这里出的性能数字没有调度意义，数值才是重点。
///
/// 只吃 token id —— 后端不带 tokenizer（那是前端的事）。文本→ids 用前端，或
/// 自备脚本按模型目录的 vocab.json + merges.txt 转换。
///
/// Usage:
///   test_model_forward <config.json> --ids 1,2,3 [--graph NAME] [--print-topk
///   5]
///                     [--json out.json]
///   test_model_forward <config.json> --ids-file req.json   # [1,2,3] 或
///   {"ids": [...]}
///
/// `--json out.json` 落盘 prompt/生成 ids 与每步（图名/耗时），交给外部脚本
/// 解码或对拍（后端不带 tokenizer）。
///
/// **所有别的参数都取自
/// config**：采样（temperature/top_p/top_k/seed/max_new_tokens）
/// 与尺寸（max_num_request/n_context/max_num_batched_tokens）都走
/// `bin::Config`，所以这里的 图与 server 完全同形，跑出来的 token 可以直接和
/// server 对拍。要换参数就换 config （或写一份 /tmp 的副本），别再加 flag。
///
/// 脚本只用 **slot 0**，其余槽留空——`step()` 要求给出整个槽池的视图（cache/mask
/// 按整池建索引，槽数必须 == max_num_request）。
///
/// `--graph NAME` = **只跑这一张图**（单图冒烟：把 ids
/// 喂完就停，看路径/耗时/logits，
/// 不接着生成）；不指定则按"静态优先"自动选图，把 prompt
/// 全部喂完（受每槽上限约束， RGD-only 的图会逐 token 喂），然后按 config
/// 的采样参数生成 max_new_tokens 个 token。

#include "../bin/config.hpp"
#include "../engine/sampler/default.hpp"
#include "../models/loader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

namespace {

// Single-slot policy for the direct-runner debug tool. Each graph's budget
// and per-slot limit must be evaluated together, as in the scheduler.
std::pair<int64_t, int64_t> pick_forward_graph(
  std::span<const finer::engine::GraphInfo> graphs, int64_t want,
  const std::string &name = {}) {
  if (want <= 0)
    throw std::invalid_argument(
      "graph selection requires positive token count");
  int64_t best = -1, take = 0;
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &g = graphs[i];
    if (!name.empty() && g.name != name)
      continue;
    if (!g.produces_logits || g.max_total_tokens <= 0)
      continue;
    auto count = std::min(want, g.max_total_tokens);
    if (g.max_tokens_per_slot > 0)
      count = std::min(count, g.max_tokens_per_slot);
    if (!name.empty() && count < want)
      throw std::runtime_error(
        "graph '" + name + "' cannot fit " + std::to_string(want) + " tokens");
    if (best < 0 || count > take
      || (count == take
        && (g.dynamic_shape != graphs[best].dynamic_shape
            ? !g.dynamic_shape
            : g.max_total_tokens < graphs[best].max_total_tokens))) {
      best = static_cast<int64_t>(i);
      take = count;
    }
  }
  if (best < 0)
    throw std::runtime_error(name.empty()
        ? "no logits graph can serve tokens"
        : "no usable logits graph named '" + name + "'");
  return {best, take};
}

using finer::engine::SlotStep;
using finer::engine::StepOutput;

struct Args {
  std::string          config;
  std::vector<int64_t> ids;
  std::string          graph; // 非空 = 只喂这张图（单图冒烟）
  int64_t              print_topk = 5; // 0 = 不打候选
  std::string          json_out; // 非空 = 结果落盘（ids + 每步）
};

[[noreturn]] void usage(const char *argv0) {
  std::cerr
    << "Usage: " << argv0
    << " <config.json> --ids 1,2,3 [options]\n"
       "  --ids A,B,C        输入 token id（逗号分隔）\n"
       "  --ids-file FILE    从 json 读： [1,2,3] 或 {\"ids\": [...]}\n"
       "  --graph NAME       "
       "只跑这一张图（单图冒烟，喂完即停），不指定则自动选图\n"
       "  --print-topk N     每步打印前 N 个候选（默认 5，0 = 不打印）\n"
       "  --json FILE        结果落盘（ids + 每步图名/耗时）\n"
       "（采样参数与尺寸都取自 config）\n";
  std::exit(1);
}

std::vector<int64_t> parse_ids(const std::string &s) {
  std::vector<int64_t> out;
  size_t               pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    const auto   piece = s.substr(pos, comma - pos);
    if (!piece.empty())
      out.push_back(std::stoll(piece));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

Args parse_args(int argc, char **argv) {
  if (argc < 2)
    usage(argv[0]);
  Args a;
  a.config = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto        next = [&]() -> std::string {
      if (i + 1 >= argc)
        usage(argv[0]);
      return argv[++i];
    };
    if (flag == "--ids")
      a.ids = parse_ids(next());
    else if (flag == "--ids-file") {
      const auto j   = nlohmann::json::parse(std::ifstream(next()));
      const auto arr = j.is_array()
        ? j
        : (j.contains("ids") ? j["ids"] : j["prompt_token_ids"]);
      for (const auto &v : arr)
        a.ids.push_back(v.get<int64_t>());
    } else if (flag == "--graph")
      a.graph = next();
    else if (flag == "--print-topk")
      a.print_topk = std::stoll(next());
    else if (flag == "--json")
      a.json_out = next();
    else
      usage(argv[0]);
  }
  if (a.ids.empty()) {
    std::cerr << "error: --ids/--ids-file 都必须给出输入\n";
    usage(argv[0]);
  }
  return a;
}

/// 打印一步的 top-k 候选（indices 是 vocab token id）。
void print_topk(const StepOutput &out, int64_t slot, int64_t n) {
  if (n <= 0 || out.vals[slot].empty())
    return;
  std::string line;
  for (int64_t i = 0; i < std::min<int64_t>(n, out.vals[slot].size()); ++i)
    line += " " + std::to_string(out.indices[slot][i]) + ":"
      + std::to_string(out.vals[slot][i]);
  spdlog::info(
    "[debug]   top{}:{}", std::min<int64_t>(n, out.vals[slot].size()), line);
}

} // namespace

int main(int argc, char **argv) {
  spdlog::cfg::load_env_levels();
  const Args args = parse_args(argc, argv);

  const auto config_json = nlohmann::json::parse(std::ifstream(args.config));
  finer::bin::Config bin_cfg(config_json);

  // 尺寸与采样参数全部取自 config（不覆盖）：图与 server 完全同形，跑出来的
  // token 可以直接和 server 对拍；prefill
  // 若超出图预算/每槽上限，下面的循环会自然地多步喂。
  const auto   &ec       = bin_cfg.engine;
  const auto   &srv      = bin_cfg.server;
  const int64_t prompt_n = static_cast<int64_t>(args.ids.size());
  // step() 的 slots 必须覆盖整个槽池（cache/mask 按池建索引）；只用 slot 0。
  const size_t slots_n = static_cast<size_t>(ec.max_num_request);
  if (prompt_n + srv.max_new_tokens > ec.n_context)
    spdlog::warn("[debug] prompt({}) + max_new_tokens({}) 超出 n_context({})，"
                 "生成会在上下文用尽处被图拒绝",
      prompt_n, srv.max_new_tokens, ec.n_context);

  spdlog::info("[debug] model={} bs={} n_context={} prefill_chunk={} prompt={} "
               "max_new={} temperature={} top_p={} top_k={} seed={}",
    bin_cfg.model.string(), ec.max_num_request, ec.n_context,
    ec.max_num_batched_tokens, prompt_n, srv.max_new_tokens, srv.temperature,
    srv.top_p, srv.top_k, srv.seed);

  if (ec.tp.world_size() != 1)
    throw std::invalid_argument("test_model_forward is single-rank; use "
                                "TP=N run-server.sh for TP validation");
  auto runner =
    finer::models::load_runner(bin_cfg.model, ec, bin_cfg.model_options);

  const auto infos = runner->graph_infos();
  for (size_t i = 0; i < infos.size(); ++i)
    spdlog::info("[debug] graph[{}] name={} budget={} per_slot_cap={} "
                 "logits={} dynamic={}",
      i, infos[i].name, infos[i].max_total_tokens, infos[i].max_tokens_per_slot,
      infos[i].produces_logits, infos[i].dynamic_shape);

  // ── 准入 ──
  const auto admit = runner->admit_request(0, args.ids, {});

  // ── prefill：把 prompt 喂完（受每槽上限约束时自然变成多步）──
  struct StepRec {
    std::string          kind, graph;
    int64_t              tokens;
    double               ms;
    std::vector<float>   topk_values;
    std::vector<int64_t> topk_indices;
  };
  std::vector<StepRec> step_recs;
  std::vector<int64_t> generated;
  auto                 write_result = [&](const std::vector<int64_t> &gen) {
    if (args.json_out.empty())
      return;
    nlohmann::json steps = nlohmann::json::array();
    for (const auto &r : step_recs)
      steps.push_back({{"kind", r.kind}, {"graph", r.graph},
        {"tokens", r.tokens}, {"ms", r.ms}, {"topk_values", r.topk_values},
        {"topk_indices", r.topk_indices}});
    nlohmann::json j = {{"model", bin_cfg.model.string()},
      {"prompt_token_ids", args.ids}, {"generated_token_ids", gen},
      {"steps", steps}};
    std::ofstream(args.json_out) << j.dump(2) << "\n";
    spdlog::info("[debug] 结果写入 {}", args.json_out);
  };

  int64_t    cursor = admit.cache_hint_len;
  StepOutput out;
  while (cursor < prompt_n) {
    const auto [gi, take] =
      pick_forward_graph(infos, prompt_n - cursor, args.graph);
    const auto                 t0 = std::chrono::steady_clock::now();
    const std::vector<int64_t> chunk(
      args.ids.begin() + cursor, args.ids.begin() + cursor + take);
    std::vector<SlotStep> steps(slots_n); // slot 0 活跃，其余空
    steps[0] = {cursor, chunk};
    if (!runner->can_run(gi, steps))
      throw std::runtime_error("single-request harness exceeds KV capacity");
    out           = runner->step(gi, steps);
    const auto ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0)
                      .count();
    spdlog::info("[debug] prefill graph={} tokens=[{}, {}) take={} {:.1f}ms",
      infos[static_cast<size_t>(gi)].name, cursor, cursor + take, take, ms);
    step_recs.push_back({"prefill", infos[static_cast<size_t>(gi)].name, take,
      ms, out.vals[0], out.indices[0]});
    cursor += take;
  }
  print_topk(out, 0, args.print_topk);
  if (!args.graph.empty()) { // 单图冒烟：喂完就停
    spdlog::info("[debug] --graph {} 单图冒烟结束（不生成）", args.graph);
    write_result(generated);
    return 0;
  }

  // ── decode：贪心（temperature<=0）或按 seed 采样，逐 token 跑 ──
  std::mt19937_64 rng(static_cast<uint64_t>(srv.seed));
  for (int64_t step = 0; step < srv.max_new_tokens; ++step) {
    auto &vals = out.vals[0];
    auto &ix   = out.indices[0];
    if (vals.empty()) {
      spdlog::warn("[debug] 该步没有 logits，停止");
      break;
    }
    int64_t next = ix[0]; // 图内 TopK 已排序 → 第 0 个是 argmax
    if (srv.temperature > 0.0F) {
      // 和 scheduler 的 sample_logits 一致：图内 TopK 已把候选排好序，CPU 只做
      // Temperature→TopP（top_k 在这里只用于截断候选数，0 = 全部）。
      const int64_t                  k = srv.top_k > 0
                         ? std::min<int64_t>(vals.size(), srv.top_k)
                         : static_cast<int64_t>(vals.size());
      std::vector<float>             keep(vals.begin(), vals.begin() + k);
      const std::vector<int64_t>     keep_ix(ix.begin(), ix.begin() + k);
      finer::sampler::Default::Param p{
        .top_k = 0, .top_p = srv.top_p, .temperature = srv.temperature};
      next = finer::sampler::random_sample(
        finer::sampler::Default(p).sample(finer::sampler::Logits(
          std::span<float>(keep), std::span<const int64_t>(keep_ix))),
        rng);
    }
    generated.push_back(next);

    const auto            gi = pick_forward_graph(infos, 1, args.graph).first;
    const auto            t0 = std::chrono::steady_clock::now();
    std::vector<SlotStep> steps(slots_n); // slot 0 活跃，其余空
    steps[0] = {cursor, std::span<const int64_t>(&generated.back(), 1)};
    if (!runner->can_run(gi, steps))
      throw std::runtime_error("single-request harness exceeds KV capacity");
    out           = runner->step(gi, steps);
    const auto ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0)
                      .count();
    spdlog::info("[debug] decode[{}] token={} graph={} {:.1f}ms", step, next,
      infos[static_cast<size_t>(gi)].name, ms);
    step_recs.push_back({"decode", infos[static_cast<size_t>(gi)].name, 1, ms,
      out.vals[0], out.indices[0]});
    print_topk(out, 0, args.print_topk);
    ++cursor;
  }

  std::string ids_line;
  for (const auto t : generated)
    ids_line += " " + std::to_string(t);
  spdlog::info("[debug] prompt ids:{}", [&] {
    std::string s;
    for (const auto t : args.ids)
      s += " " + std::to_string(t);
    return s;
  }());
  spdlog::info("[debug] generated ({} tokens):{}", generated.size(), ids_line);
  write_result(generated);
  return 0;
}
