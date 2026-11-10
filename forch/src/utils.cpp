
#include "forch/utils.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

namespace forch::utils {

struct IntervalTimer::Impl {
  Impl(std::string entity)
      : event(std::move(entity)), start(std::chrono::steady_clock::now()) {
    logger()->info("{} ...", this->event);
  }

  static std::string format_duration(std::chrono::steady_clock::duration d) {
    using namespace std::chrono;
    const auto ns = duration_cast<nanoseconds>(d).count();

    const long long NS_PER_SEC = 1000000000LL;
    const long long NS_PER_MS  = 1000000LL;
    const long long NS_PER_US  = 1000LL;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(5);

    if (ns >= NS_PER_SEC) {
      double total_seconds = static_cast<double>(ns) / 1e9;
      int    days          = static_cast<int>(total_seconds / 86400);
      total_seconds -= static_cast<double>(days) * 86400.0;
      int hours = static_cast<int>(total_seconds / 3600);
      total_seconds -= static_cast<double>(hours) * 3600.0;
      int minutes = static_cast<int>(total_seconds / 60);
      total_seconds -= static_cast<double>(minutes) * 60.0;
      double seconds = total_seconds;

      if (days > 0) {
        oss << days << "d ";
      }
      if (hours > 0) {
        oss << hours << "h ";
      }
      if (minutes > 0) {
        oss << minutes << "m ";
      }
      oss << seconds << "s";
      return oss.str();
    }

    if (ns >= NS_PER_MS) {
      double ms = static_cast<double>(ns) / 1e6;
      oss << ms << " ms";
      return oss.str();
    }

    if (ns >= NS_PER_US) {
      double us = static_cast<double>(ns) / 1e3;
      oss << us << " us";
      return oss.str();
    }

    // Very small durations: keep integer nanoseconds
    oss.unsetf(std::ios::fixed);
    oss.str("");
    oss.clear();
    oss << ns << " ns";
    return oss.str();
  }

  void dump_now() const {
    const auto now   = std::chrono::steady_clock::now();
    const auto human = format_duration(now - start);
    logger()->info("{}: {}", event, human);
  }

  std::string                           event;
  std::chrono::steady_clock::time_point start;
};

IntervalTimer::IntervalTimer(std::string entity)
    : _impl(std::make_unique<Impl>(std::move(entity))) {}

IntervalTimer::~IntervalTimer() {
  _impl->dump_now();
}

} // namespace forch::utils
