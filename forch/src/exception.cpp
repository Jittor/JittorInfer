#include "forch/exception.hpp"

#include <backward.hpp>
#include <sstream>

namespace forch {

exception::exception(std::string message) : message(std::move(message)) {
  backward::StackTrace st;
  st.load_here(32);
  st.skip_n_firsts(3);
  backward::Printer  p;
  std::ostringstream oss;
  p.print(st, oss);
  this->message += "\nStack trace:\n" + oss.str();
}

const char *exception::what() const noexcept {
  return message.c_str();
}

}; // namespace forch