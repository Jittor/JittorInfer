#pragma once

#include <exception>
#include <string>

namespace forch {

struct exception : public std::exception {
  exception(std::string);
  const char *what() const noexcept override;

private:
  std::string message;
};

} // namespace forch