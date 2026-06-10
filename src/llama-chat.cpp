#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>

#include "llama.h"

#if __cplusplus >= 202000L
#    define LU8(x) (const char *) (u8##x)
#else
#    define LU8(x) u8##x
#endif

enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    LLM_CHAT_TEMPLATE_MISTRAL_V1,
    LLM_CHAT_TEMPLATE_MISTRAL_V3,
    LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    LLM_CHAT_TEMPLATE_MISTRAL_V7,
    LLM_CHAT_TEMPLATE_PHI_3,
    LLM_CHAT_TEMPLATE_PHI_4,
    LLM_CHAT_TEMPLATE_FALCON_3,
    LLM_CHAT_TEMPLATE_ZEPHYR,
    LLM_CHAT_TEMPLATE_MONARCH,
    LLM_CHAT_TEMPLATE_GEMMA,
    LLM_CHAT_TEMPLATE_ORION,
    LLM_CHAT_TEMPLATE_OPENCHAT,
    LLM_CHAT_TEMPLATE_VICUNA,
    LLM_CHAT_TEMPLATE_VICUNA_ORCA,
    LLM_CHAT_TEMPLATE_DEEPSEEK,
    LLM_CHAT_TEMPLATE_DEEPSEEK_2,
    LLM_CHAT_TEMPLATE_DEEPSEEK_3,
    LLM_CHAT_TEMPLATE_QWEN2,
    LLM_CHAT_TEMPLATE_QWEN3,
    LLM_CHAT_TEMPLATE_COMMAND_R,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_CHATGML_3,
    LLM_CHAT_TEMPLATE_CHATGML_4,
    LLM_CHAT_TEMPLATE_GLMEDGE,
    LLM_CHAT_TEMPLATE_MINICPM,
    LLM_CHAT_TEMPLATE_EXAONE_3,
    LLM_CHAT_TEMPLATE_RWKV_WORLD,
    LLM_CHAT_TEMPLATE_GRANITE,
    LLM_CHAT_TEMPLATE_GIGACHAT,
    LLM_CHAT_TEMPLATE_MEGREZ,
    LLM_CHAT_TEMPLATE_UNKNOWN,
};

static const std::map<std::string, llm_chat_template> LLM_CHAT_TEMPLATES = {
    { "chatml",            LLM_CHAT_TEMPLATE_CHATML            },
    { "llama2",            LLM_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        LLM_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        LLM_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        LLM_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        LLM_CHAT_TEMPLATE_MISTRAL_V7        },
    { "phi3",              LLM_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              LLM_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           LLM_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            LLM_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           LLM_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             LLM_CHAT_TEMPLATE_GEMMA             },
    { "orion",             LLM_CHAT_TEMPLATE_ORION             },
    { "openchat",          LLM_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            LLM_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       LLM_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          LLM_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         LLM_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         LLM_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "qwen2",             LLM_CHAT_TEMPLATE_QWEN2             },
    { "qwen3",             LLM_CHAT_TEMPLATE_QWEN3             },
    { "qwen3moe",          LLM_CHAT_TEMPLATE_QWEN3             },
    { "command-r",         LLM_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            LLM_CHAT_TEMPLATE_LLAMA_3           },
    { "chatglm3",          LLM_CHAT_TEMPLATE_CHATGML_3         },
    { "chatglm4",          LLM_CHAT_TEMPLATE_CHATGML_4         },
    { "glmedge",           LLM_CHAT_TEMPLATE_GLMEDGE           },
    { "minicpm",           LLM_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           LLM_CHAT_TEMPLATE_EXAONE_3          },
    { "rwkv-world",        LLM_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           LLM_CHAT_TEMPLATE_GRANITE           },
    { "gigachat",          LLM_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            LLM_CHAT_TEMPLATE_MEGREZ            },
};

static llm_chat_template llm_chat_template_from_str(const std::string & name) {
    return LLM_CHAT_TEMPLATES.at(name);
}

static llm_chat_template llm_chat_detect_template(const std::string & tmpl) {
    if (LLM_CHAT_TEMPLATES.count(tmpl) > 0) {
        return LLM_CHAT_TEMPLATES.at(tmpl);
    }

    auto tmpl_contains = [&tmpl](const char * haystack) -> bool {
        return tmpl.find(haystack) != std::string::npos;
    };
    if (tmpl_contains("'Assistant: ' + message['content'] + eos_token")) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_2;
    }
    if (tmpl_contains("<|im_start|>") && tmpl_contains("<|im_end|>")) {
        // 识别到 ChatML/Jinja 风格，使用 ChatML 模板，Qwen2.5 和 Qwen3 都使用这个模板
        return LLM_CHAT_TEMPLATE_CHATML;
    }
    if (tmpl_contains("Assistant:") && tmpl_contains("User:")) {
        // 简单启发式：若传入的是类似 DeepSeek/Qwen2 的纯文本模板，则回落到 QWEN2 格式
        return LLM_CHAT_TEMPLATE_QWEN2;
    }
    { GGML_ABORT("Unknown template"); }
}

// Simple version of "llama_apply_chat_template" that only works with strings
// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
static int32_t llm_chat_apply_template(llm_chat_template tmpl, const std::vector<const llama_chat_message *> & chat,
                                       std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggerganov/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK_2) {
        // DeepSeek-V2
        for (const auto * message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_QWEN2) {
        // Qwen2：按简单的“User/Assistant”文本格式拼接（不引入 DeepSeek 的特殊结束 token）
        for (const auto * message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content;
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
        for (const auto * message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}

//
// chat templates
//

int32_t llama_chat_apply_template(const char * tmpl, const struct llama_chat_message * chat, size_t n_msg, bool add_ass,
                                  char * buf, int32_t length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string       formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}
