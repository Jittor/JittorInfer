#include "forch-ge-common.hpp"

#include "forch/exception.hpp"
#include "magic_enum.hpp"

#include <array>
#include <string>

namespace forch::detail {

namespace {

const auto c_dtype_mapping = [] {
  std::array<CDtypeDesc, magic_enum::enum_count<DType>()> mapping{};
  mapping[static_cast<size_t>(DType::FP64)]  = {C_DT_DOUBLE, 8};
  mapping[static_cast<size_t>(DType::FP32)]  = {C_DT_FLOAT, 4};
  mapping[static_cast<size_t>(DType::FP16)]  = {C_DT_FLOAT16, 2};
  mapping[static_cast<size_t>(DType::BF16)]  = {C_DT_BF16, 2};
  mapping[static_cast<size_t>(DType::INT32)] = {C_DT_INT32, 4};
  mapping[static_cast<size_t>(DType::INT64)] = {C_DT_INT64, 8};
  mapping[static_cast<size_t>(DType::INT8)]  = {C_DT_INT8, 1};
  mapping[static_cast<size_t>(DType::BOOL)]  = {C_DT_BOOL, 1};
  return mapping;
}();

} // namespace

void shim_check(int rc, const char *what) {
  if (rc != 0) {
    throw exception(std::string("forch-ge: ") + what + "\n== shim error ==\n"
      + forch_ge_last_error());
  }
}

CDtypeDesc::CDtypeDesc(C_DataType type, size_t size) : type(type), size(size) {}

CDtypeDesc::CDtypeDesc(DType dtype) {
  *this = c_dtype_mapping.at(static_cast<size_t>(dtype));
}

DType c_to_forch_dtype(C_DataType dtype) {
  for (size_t i = 0; i < c_dtype_mapping.size(); ++i) {
    if (c_dtype_mapping[i].type == dtype) {
      return static_cast<DType>(i);
    }
  }
  throw exception(
    "forch-ge: unsupported CANN data type: " + std::to_string(dtype));
}

} // namespace forch::detail
