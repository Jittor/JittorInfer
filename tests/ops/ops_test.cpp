#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "ops_test_case.h"
#include <cstdint>
#include <iostream>
#include <ostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

const int max_graph_nodes = 32;
const float abs_error_bar = 0.01;
const float rel_error_bar = 0.002;

void ops_test_cpu(test_case* op_case, std::vector<float>& output){
// 初始化输出向量
    output.resize(op_case->output_size, 0.0f);\
    struct ggml_init_params init_params = {
        /* .mem_size */   ggml_tensor_overhead() * 32 + 1024 * 1024 * 128,  // 分配足够的内存
        /* .mem_buffer */ NULL,
        /* .no_alloc */   false
    };
    
    // 创建GGML上下文
    ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }
    
    // 创建输入张量并复制数据
    std::vector<ggml_tensor*> src_tensors;
    for(auto &val : op_case->src){
        ggml_tensor* tmp = ggml_new_tensor(ctx, val->type, 4 , val->ne);
        memcpy(tmp->data, val->data, ggml_nbytes(val));
        src_tensors.push_back(tmp);
    }
    // 构建计算图
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 100, false);
    
    ggml_tensor* output_tensor = op_case->build_graph_cpu(ctx, src_tensors);
    
    // ggml_tensor* output_tensor = ggml_add(ctx, src_tensors[0], src_tensors[1]);

    // 添加到计算图中
    ggml_build_forward_expand(gf, output_tensor);
    
    // 执行计算
    // ggml_build_forward_expand(gf, result); // why ?
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
    
    // 为 cplan 分配 work_data
    if (cplan.work_size > 0) {
        void* work_memory = malloc(cplan.work_size);
        if (!work_memory) {
            std::cerr << "Failed to allocate work memory for cplan" << std::endl;
            ggml_free(ctx);
            return;
        }
        cplan.work_data = (uint8_t*)work_memory;
    }
    
    ggml_status status = ggml_graph_compute(gf, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("attention_cpu failed with status : %d\n", status);
        GGML_ABORT("attention_cpu failed");
    }
    
    // 释放 work_data
    if (cplan.work_size > 0 && cplan.work_data) {
        free(cplan.work_data);
    }
    // 复制结果到输出向量
    memcpy(output.data(), output_tensor->data, output.size() * sizeof(float));
    op_case->special_check_cpu(output_tensor);
    // 释放资源
    ggml_free(ctx);
}
void build_ops_graph(
    test_case* op_case,
    std::vector<float> &output_host,
    ggml_backend_t backend){
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * max_graph_nodes + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    // build graph

    std::vector<ggml_tensor*> src_tensors;
    for(auto &val:op_case->src){
        src_tensors.push_back(ggml_new_tensor(ctx, val->type, 4 , val->ne));
    }
    
    ggml_tensor* output_tensor = op_case->build_graph(ctx, src_tensors);
    GGML_ASSERT(ggml_nelements(output_tensor) == op_case->output_size);

    ggml_build_forward_expand(gf, output_tensor);

    // 分配空间
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    // 设置输入
    for(int i = 0; i < op_case->src.size(); ++i){
        GGML_ASSERT(ggml_nbytes(src_tensors[i]) == ggml_nbytes(op_case->src[i]));
        ggml_backend_tensor_set(src_tensors[i], op_case->src[i]->data, 0, ggml_nbytes(src_tensors[i]));
    }
    
    // 执行计算
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    // 获取输出
    output_host.resize(ggml_nelements(output_tensor));
    ggml_backend_tensor_get(output_tensor, output_host.data(), 0, ggml_nbytes(output_tensor));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}
bool calculate_error(std::vector<float> &output_host,std::vector<float> &output_host_cann, int64_t output_size){
    // 比较 CPU 和 CANN 实现的结果
    std::cout << "Comparing CPU and CANN implementations:" << std::endl;
    
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int max_abs_error_idx = 0;
    int max_rel_error_idx = 0;
    
    for (int64_t i = 0; i < output_size; ++i) {
        double abs_error = std::fabs(output_host[i] - output_host_cann[i]);
        double rel_error = 0.0;
        
        // 计算相对误差，避免除以0
        if (std::fabs(output_host[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host[i]);
        } else if (std::fabs(output_host_cann[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host_cann[i]);
        }
        
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_abs_error_idx = i;
        }
        
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
            max_rel_error_idx = i;
        }
    }
    
    std::cout << "Maximum absolute error: " << max_abs_error 
              << " at index " << max_abs_error_idx 
              << " (CPU: " << std::fixed << std::setprecision(9) << output_host[max_abs_error_idx] 
              << ", CANN: " << std::fixed << std::setprecision(9) << output_host_cann[max_abs_error_idx] << ")" << std::endl;
              
    std::cout << "Maximum relative error: " << max_rel_error 
              << " at index " << max_rel_error_idx 
              << " (CPU: " << std::fixed << std::setprecision(9) << output_host[max_rel_error_idx] 
              << ", CANN: " << std::fixed << std::setprecision(9) << output_host_cann[max_rel_error_idx] << ")" << std::endl;
    return (max_abs_error <= abs_error_bar) && (max_rel_error <= rel_error_bar);
}
bool ops_test(ggml_backend_dev_t cann_dev, test_case* op_case){
    op_case->init_src_size();
    std::cout << "Initializing tensors with random values..." << std::endl;
    op_case->init_tensors();
    std::vector<float> output_host_cpu(op_case->output_size);

    ops_test_cpu(op_case, output_host_cpu);

    std::cout << "Cpu output tensor sample (first few values):" << std::endl;
    for (int i = 0; i < std::min(static_cast<int64_t>(10), op_case->output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host_cpu[i]) << " ";
    }
    std::cout << std::endl;

    std::vector<float> output_host_cann(op_case->output_size);
    // 创建后端
    ggml_backend_t cann_backend = ggml_backend_dev_init(cann_dev, NULL);
    GGML_ASSERT(cann_backend != NULL);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cann_dev);

    build_ops_graph(op_case, output_host_cann, cann_backend);
    GGML_ASSERT(output_host_cann.size() == op_case->output_size);
    std::cout << "Cann output tensor sample (first few values):" << std::endl;
    for (int i = 0; i < std::min(static_cast<int64_t>(10), op_case->output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host_cann[i]) << " ";
    }
    std::cout << std::endl;

    bool accepted = calculate_error(output_host_cpu, output_host_cann, op_case->output_size);

    return accepted;
}

int main() {
    std::cout<<"Finding Decive:"<<std::endl;
    int num_devices = ggml_backend_dev_count();

    ggml_backend_dev_t cann_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::cout << "Device " << i << " name: " << ggml_backend_dev_name(dev) << std::endl;
        if (std::string(ggml_backend_dev_name(dev)).find("CANN") != std::string::npos) {
            std::cout << "CANN device found" << std::endl;
            cann_dev = dev;
            break;
        }
    }
    if (cann_dev == nullptr) {
        std::cerr << "CANN device not found" << std::endl;
        return 1;
    }
    printf("  Device description: %s\n", ggml_backend_dev_description(cann_dev));
    size_t free, total;  // NOLINT
    ggml_backend_dev_memory(cann_dev, &free, &total);
    printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
    printf("\n");
    std::vector<test_case*> test_cases = {
        new test_case_add(),
        // new test_case_sub(), unimplemented
        new test_case_mul(),
        new test_case_div(),
        new test_case_sqr(),
        new test_case_sum_rows(),
        new test_case_acc(),
        new test_case_norm(),
        new test_case_group_norm(),
        new test_case_concat(),
        new test_case_upscale(),
        new test_case_pad(),
        new test_case_arange(),
        new test_case_gelu(),
        new test_case_gelu_quick(),
        new test_case_silu(),
        new test_case_tanh(),
        new test_case_relu(),
        new test_case_hardsigmoid(),
        new test_case_hardswish(),
        new test_case_timestep_embedding(),
        new test_case_rms_norm_fused(),
        new test_case_leaky_relu(),
        new test_case_rms_norm(),
        new test_case_mul_mat(),
        new test_case_mul_mat_id(),
        new test_case_scale(),
        new test_case_clamp(),
        new test_case_cpy(),
        new test_case_cont(),
        new test_case_diag_mask_inf(),
        new test_case_soft_max(),
        new test_case_soft_max({64,64,1,1},{5,5},true), // with mask
        new test_case_rope(), // fails when n_dim is not power of 2
        new test_case_im2col(),
        new test_case_pool_2d(),
        new test_case_pool_2d({32,32,5,1},GGML_OP_POOL_MAX),
        new test_case_argsort(),
        new test_case_argsort({32,32,5,1},GGML_SORT_ORDER_DESC),
        new test_case_to_zero(),
        // new test_case_all_reduce_sum(), // unimplemented
        new test_case_get_slice(),
        // new test_case_scatter_update() // with strange bug ?
        new test_case_moe_fused(),
        new test_case_flash_attn_jittor(),
        new test_case_flash_attn_prompt()
    };
    std::vector<std::string> failed_operation; 
    for (int i = 0; i < test_cases.size(); ++i){
        std::cout<<"testing case [" << i <<"]: operation :["<<test_cases[i]-> name<<"]" << std::endl;
        bool res = ops_test(cann_dev, test_cases[i]);
        if (res){
            std::cout << "Accepted!" << std::endl;
        }else{
            std::cout << "Failed!" <<std::endl;
            failed_operation.push_back(test_cases[i]->name);
        }
    }
    std::cout << "Test result : " << failed_operation.size() << " / " << test_cases.size() << " failed :" << std::endl;
    for(auto &val : failed_operation){
        std::cout << val << std::endl;
    }
    return 0;
}
