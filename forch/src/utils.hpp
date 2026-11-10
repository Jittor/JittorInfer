#pragma once

// Private, backend-independent implementation helpers. Not installed.
#include "forch/exception.hpp"

#include <charconv>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>

namespace forch::detail {

/**
 * Read a decimal integer, using fallback for an unset or empty variable.
 * Reject overflow, whitespace and partial parses: silently truncating a rank
 * or device id can send different processes to the same participant/device.
 */
template <class T>
T getenv_int(const char *name, T fallback) {
  static_assert(std::is_integral_v<T>);
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  const std::string_view text(raw);
  T                      value{};
  const auto [end, error] =
    std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw exception(std::string(name)
      + ": expected an in-range decimal integer, got '" + raw + "'");
  }
  return value;
}

} // namespace forch::detail
