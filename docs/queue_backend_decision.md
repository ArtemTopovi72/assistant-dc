# Queue backend: why Redis is in the request path and Kafka is not

## The complaint

Users reported the bot's queue behaving badly and people getting blocked behind
each other. The queue code was never the first suspect — until it turned out
that `KAFKA_BROKERS` and `REDIS_URL` were both unset, so `_make_backend()` had
been silently selecting `InMemoryBackend` on every start since the Redis and
Kafka backends were written. Neither had ever executed in production.

Both services are now installed and both backends are exercised against them.
This document records which one carries production traffic, and why.

## The infrastructure (both installed, both kept)

| | Version | Endpoint | Lifecycle |
|---|---|---|---|
| Redis | Memurai Developer 4.1.2 (Redis 7.2.5) | `127.0.0.1:6379` | Windows service, auto-start |
| Kafka | Apache Kafka 3.9.1, KRaft (no Zookeeper) | `127.0.0.1:9092` (+9093 controller) | scheduled task at logon |
| Java | Microsoft OpenJDK 21.0.12 | — | for Kafka only |

`REDIS_URL` is set at user scope. `KAFKA_BROKERS` is deliberately **not** set —
see the measurements below. Kafka stays installed, running and covered by
`tests/test_tg_queue_backends.py` as infrastructure, so enabling it later is one
environment variable, not a reinstall.

## The measurements

Both backends driven through their real `push`/`pop` against the real services.

### Fairness — one heavy user floods, two light users queue behind

Chat 100 queues 10 jobs; chats 200 and 300 then queue 1 each.

| Backend | light200 served at | light300 served at |
|---|---|---|
| **Redis** | **position 2** | **position 3** |
| Kafka | position 11 | position 12 |

Redis order: `heavy0, light200, light300, heavy1, heavy2, …`
Kafka order: `heavy0 … heavy9, light200, light300`

### Starvation — light user arrives mid-flood

Chat 100 queues 6, chat 200 queues 1, chat 100 queues 6 more.

| Backend | B0 served at |
|---|---|
| **Redis** | **position 2 of 13** |
| Kafka | position 7 of 13 |

### Throughput — 200 tasks across 8 chats

| Backend | push | drain |
|---|---|---|
| **Redis** | 18,705/s | 9,987/s |
| Kafka | 1,081/s | 3,996/s |

### Worker death — task in flight when the worker dies

Both backends lose the popped task: Redis `LPOP`s without an ack, Kafka
auto-commits. **This is handled a layer up, not in the backend.**
`TaskQueue._write_inflight()` journals running and pending tasks, and
`_recover_inflight()` notifies the affected user on restart and offers a retry.
It correctly treats Redis and Kafka as durable — a task still sitting in the
queue is popped and run normally by the fresh consumer, so it is not
double-reported as "interrupted".

## The decision

**Redis carries production traffic.** It wins on every axis that maps to the
original complaint: light users are served 5x sooner under load, it does not
starve anyone behind a flood, and it is an order of magnitude faster to enqueue.

**Kafka is not in the request path.** Not because it is broken — it passes its
tests — but because `KafkaBackend` is FIFO by construction. Its own docstring
says so: Kafka partitions are consumed in order and offsets cannot be reordered
client-side, so the per-chat round-robin that `InMemoryBackend` and
`RedisBackend` implement cannot be expressed. Putting Kafka in the request path
means one user with ten queued jobs blocks everyone behind them — which is
precisely the reported problem.

Kafka would be the right choice if the requirement changed to multi-process
consumers, cross-host fan-out, or replay from an offset. It is installed and
tested so that switch stays cheap.

## Operational notes

- **Never `--delete` a Kafka topic on this host.** Windows cannot unlink the
  memory-mapped log segments; the delete marks the log dir failed and the broker
  shuts down (`Shutdown broker because all log dirs in C:\kafka-logs have
  failed`). Observed directly. Recreate by wiping `C:\kafka-logs` and
  re-running `kafka-storage.bat format`.
- **`tgbot_tasks` must stay single-partition.** `KafkaBackend` produces without
  a key, so with the broker default `num.partitions=3` messages scatter and come
  back out of order — pushed `k1,k2`, popped `k2,k1`. `num.partitions=1` is set
  in `config/kraft/local.properties`.
- **Redis must be 6.0.6 or newer.** `RedisBackend`'s pop script uses `LPOS`.
  The common Windows port is Redis 5.0.14, where the client connects and `PING`s
  successfully and then the Lua fails at runtime. The suite asserts the server
  version so this reports as "server too old" rather than an opaque Lua error.
- Kafka needs a short install path (`C:\kafka`). At a longer path the generated
  CLASSPATH exceeds cmd's 8191-character limit and every `.bat` fails with
  "The input line is too long."
