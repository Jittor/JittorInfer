#include "chat_local.h"

#include <optional>

#include "chat-template.hpp"
#include "json-schema-to-grammar_local.h"
#include "minja.hpp"

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
    bool has_explicit_template;  // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

struct templates_params {
    json        messages;
    json        tools;
    json        json_schema;
    bool        stream;
    std::string grammar;
    bool        add_generation_prompt = true;
};

template <> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::runtime_error("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::runtime_error("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::runtime_error("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            if (message.contains("content")) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::runtime_error("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text") {
                            throw std::runtime_error("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::runtime_error("Invalid 'content' type: expected string or array, got " + content.dump() +
                                             " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            } else {
                throw std::runtime_error("Expected 'content' (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()) +
                                 "; messages = " + messages.dump(2));
    }

    return msgs;
}

template <> json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    json messages = json::array();
    for (const auto & msg : msgs) {
        if (!msg.content.empty() && !msg.content_parts.empty()) {
            throw std::runtime_error("Cannot specify both content and content_parts");
        }
        json jmsg{
            { "role", msg.role },
        };
        if (!msg.content.empty()) {
            jmsg["content"] = msg.content;
        } else if (!msg.content_parts.empty()) {
            if (concat_typed_text) {
                std::string text;
                for (const auto & part : msg.content_parts) {
                    if (part.type != "text") {
                        continue;
                    }
                    if (!text.empty()) {
                        text += '\n';
                    }
                    text += part.text;
                }
                jmsg["content"] = text;
            } else {
                auto & parts = jmsg["content"] = json::array();
                for (const auto & part : msg.content_parts) {
                    parts.push_back({
                        { "type", part.type },
                        { "text", part.text },
                    });
                }
            }
        } else {
            jmsg["content"] = json();  // null
        }
        if (!msg.reasoning_content.empty()) {
            jmsg["reasoning_content"] = msg.reasoning_content;
        }
        if (!msg.tool_name.empty()) {
            jmsg["name"] = msg.tool_name;
        }
        messages.push_back(jmsg);
    }
    return messages;
}

template <> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const std::string & messages) {
    return common_chat_msgs_parse_oaicompat(json::parse(messages));
}

template <> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::runtime_error("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::runtime_error("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::runtime_error("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::runtime_error("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.at("description"),
                    /* .parameters = */ function.at("parameters").dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

template <> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const std::string & tools) {
    return common_chat_tools_parse_oaicompat(json::parse(tools));
}

template <> json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function",
             {
                  { "name", tool.name },
                  { "description", tool.description },
                  { "parameters", json::parse(tool.parameters) },
              }                      },
        });
    }
    return result;
}

std::string common_chat_format_example(const struct common_chat_templates * tmpls) {
    common_chat_templates_inputs inputs;
    auto                         add_simple_msg = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }
    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    try {
        tmpls->template_default = std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        tmpls->template_default = std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, token_bos, token_eos);
    }
    if (!template_tool_use_src.empty()) {
        tmpls->template_tool_use = std::make_unique<minja::chat_template>(template_tool_use_src, token_bos, token_eos);
    }
    return tmpls;
}

const common_grammar_options grammar_options{
    /* .dotall = */ false,
    /* .compact_spaces = */ false,
    // /* .compact_spaces = */ true,
};

static bool parse_json(std::string::const_iterator & it, const std::string::const_iterator & end, json & out) {
    // // https://json.nlohmann.me/features/parsing/sax_interface/
    struct json_error_locator : public nlohmann::json_sax<json> {
        std::size_t position;
        bool        found_error;

        json_error_locator() : position(0), found_error(false) {}

        bool parse_error(std::size_t position, const std::string &, const json::exception &) override {  // NOLINT
            this->position    = position - 1;
            this->found_error = true;
            return false;
        }

        bool null() override { return true; }                                          // NOLINT

        bool boolean(bool) override { return true; }                                   // NOLINT

        bool number_integer(number_integer_t) override { return true; }                // NOLINT

        bool number_unsigned(number_unsigned_t) override { return true; }              // NOLINT

        bool number_float(number_float_t, const string_t &) override { return true; }  // NOLINT

        bool string(string_t &) override { return true; }                              // NOLINT

        bool binary(binary_t &) override { return true; }                              // NOLINT

        bool start_object(std::size_t) override { return true; }                       // NOLINT

        bool key(string_t &) override { return true; }                                 // NOLINT

        bool end_object() override { return true; }

        bool start_array(std::size_t) override { return true; }  // NOLINT

        bool end_array() override { return true; }
    };

    json_error_locator err_loc;
    json::sax_parse(it, end, &err_loc);

    std::string::const_iterator temptative_end;
    if (err_loc.found_error) {
        temptative_end = it + err_loc.position;
    } else {
        temptative_end = end;
    }
    std::string json_sub{ it, temptative_end };
    try {
        out = json::parse(json_sub);
        it  = temptative_end;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

static std::string apply(const common_chat_template & tmpl, const nlohmann::ordered_json & messages,
                         const nlohmann::ordered_json & tools, bool add_generation_prompt,
                         const nlohmann::ordered_json & extra_context = nlohmann::ordered_json()) {
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages              = messages;
    tmpl_inputs.tools                 = tools;
    tmpl_inputs.add_generation_prompt = add_generation_prompt;
    tmpl_inputs.extra_context         = extra_context;
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto                         result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

static common_chat_params common_chat_params_init_generic(const common_chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;

    auto tool_call_schemas = json::array();
    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & function    = tool.at("function");
        auto         tool_schema = json{
                    { "type",       "object"                             },
                    { "properties",
                     {
                  { "name",
                            {
                        { "type", "string" },
                        { "const", function.at("name") },
                    } },
                  { "arguments", function.at("parameters") },
              }                                                          },
                    { "required",   json::array({ "name", "arguments" }) },
        };
        if (function.contains("description")) {
            tool_schema["description"] = function.at("description");
        }
        tool_call_schemas.emplace_back(tool_schema);
    });
    const auto tool_call = json{
        { "type",       "object"                     },
        { "properties",
         {
              { "tool_call", tool_call_schemas.size() == 1 ? tool_call_schemas[0] :
                                                             json{
                                                                 { "anyOf", tool_call_schemas },
                                                             } },
          }                                          },
        { "required",   json::array({ "tool_call" }) },
    };
    const auto schema = tool_call;

    data.grammar_lazy = false;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) { builder.add_schema("root", schema); },
                                 grammar_options);

    auto tweaked_messages =
        common_chat_template::add_system(inputs.messages,
                                         "Respond in JSON format, either with `tool_call` (a request to call tools) or "
                                         "with `response` reply to the user's request");

    data.prompt =
        apply(tmpl, tweaked_messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    return data;
}

static common_chat_params common_chat_params_init_without_tools(const common_chat_template &    tmpl,
                                                                const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt =
        apply(tmpl, inputs.messages, inputs.tools.empty() ? json() : inputs.tools, inputs.add_generation_prompt);
    data.format       = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    data.grammar_lazy = false;
    if (!inputs.json_schema.is_null()) {
        if (!inputs.grammar.empty()) {
            throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
        }
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else {
        data.grammar = inputs.grammar;
    }
    return data;
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    templates_params params;
    params.tools = common_chat_tools_to_json_oaicompat<json>(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    params.messages = common_chat_msgs_to_json_oaicompat<json>(
        inputs.messages, /* concat_text= */ !tmpl.original_caps().requires_typed_content);
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.grammar               = inputs.grammar;
    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    if (params.tools.is_null()) {
        return common_chat_params_init_without_tools(tmpl, params);
    }

    return common_chat_params_init_generic(tmpl, params);
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    //   GGML_ASSERT(tmpls != nullptr);
    return common_chat_templates_apply_jinja(tmpls, inputs);
}

static common_chat_msg common_chat_parse_content_only(const std::string & input) {
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = input;
    return msg;
}

common_chat_msg common_chat_parse(const std::string & input, common_chat_format format) {
    return common_chat_parse_content_only(input);
}
