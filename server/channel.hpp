#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace channel {

template <class T, class Queue = std::deque<T>>
struct mpsc {

protected:
  Queue                   queue;
  std::mutex              mutex;
  std::condition_variable condition;
  bool                    _is_open = true;

public:
  template <class... Args>
  void emplace(Args &&...args) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      queue.emplace_back(std::forward<Args>(args)...);
    }
    condition.notify_one();
  }

  void push(const T &item) {
    emplace(item);
  }
  void push(T &&item) {
    emplace(std::move(item));
  }

  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return !queue.empty() || !_is_open; });
    if (!queue.empty()) {
      T item = std::move(queue.front());
      queue.pop_front();
      return item;
    } else {
      return std::nullopt;
    }
  }
  
  std::optional<T> try_pop() {
    std::unique_lock<std::mutex> lock(mutex);
    if (_is_open && !queue.empty()) {
      T item = std::move(queue.front());
      queue.pop_front();
      return item;
    } else {
      return std::nullopt;
    }
  }

  void close() {
    mutex.lock();
    _is_open = false;
    mutex.unlock();
    condition.notify_all();
  }

  // iterator-like interface
  struct iterator_fake_end {};
  struct iterator {
    mpsc            *parent;
    std::optional<T> current;

    iterator(mpsc *p) : parent(p), current(std::nullopt) {}

    iterator &operator++() {
      current = parent->pop();
      return *this;
    }

    T operator*() {
      return std::move(*current);
    }

    bool operator!=(const iterator_fake_end & /*rhs*/) const {
      return current.has_value();
    }
  };

  iterator begin() {
    return ++iterator{this};
  }

  iterator_fake_end end() {
    return {};
  }
};

} // namespace channel