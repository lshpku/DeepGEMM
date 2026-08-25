# 开发文档

## 安装

```bash
# 注意要用虚拟环境，不要覆盖主机的原版 deep_gemm
source env3.12/bin/activate

# JIT需要用的头文件
ln -sfn "$(pwd)/third-party/cutlass/include/cutlass" deep_gemm/include/
ln -sfn "$(pwd)/third-party/cutlass/include/cute" deep_gemm/include/

rm -rf build dist *.egg-info
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/deep_gemm_cpp-*.whl
```

## 单测

```bash
source env3.12/bin/activate
python tests_overlap/test_gemm_baseline.py
python tests_overlap/test_chunk.py --arrival ready
python tests_overlap/test_chunk.py --arrival cpu
python tests_overlap/test_chunk.py --arrival gpu
python tests_overlap/test_chunk.py --check-signal
```

`test_gemm_baseline.py`：对比group_gemm和chunk的性能测试，一个是调用单次group_gemm，一个是分chunk调用，实测性能差距很小，chunk方案仅慢2%，说明分chunk几乎不影响性能

`test_chunk.py`：chunk 流式全链路（gateup, swiglu, down, signal）的正确性测试。按真实路由构造布局（默认 16384 个不重复 token，topk=8，每专家区域向 128 对齐，所以有真的 padding 行），先在异步流上把全部 kernel 发出去让它们卡在 spin-wait 上，最后与 group_gemm 逐位比对 o1/o2/o3（只比真实行）并检查 `token_done` 每个 token 恰好被记 topk 次。参数：
* `--arrival ready`：任务队列放显存且开始时全部 `ready`，测纯计算/纯 spin 开销（launch 1.9ms，compute 18.4ms，total 20.0ms）
* `--arrival cpu`：任务队列放 pinned host memory（`CUDAPinnedPlace`），由 CPU 用 ctypes 直接写 `ready`，间隔 `--interval-ms`（默认 2ms）；GPU 确实按 CPU 的节奏推进
* `--arrival gpu`：任务队列放显存，由一个单 SM 的 producer kernel（`deep_gemm.simulate_chunk_arrival`，站位真实通信 kernel）在另一条流上写 `ready`；这条路径才是最终形态，spin 开销落在噪声里（total 98.7ms ≈ arrival 82ms + 计算尾巴），比 pinned 版本快约 10 倍
* `--check-signal`：每个 chunk 后同步一次，逐个 chunk 比对 `token_done` 与 CPU 算的期望值（41 个 chunk 全部吻合，padding 行 1152 行从不被 signal）；因为要逐 chunk 同步，只能配合 `--arrival ready`
* 另有 `--interval-ms` / `--num-tokens` / `--topk` 可调，`--interval-ms 0` 即测纯 spin 开销


## 开发进展

8.25: 完成最简 demo，验证领任务机制的正确性
* 新增 `GemmType::MGroupedChunk`，复用原 `sm100_bf16_gemm_impl`，只是调度器改为在 kernel 开头从任务队列取 `(expert_idx, m_start, m_size)`，据此决定 M 方向的 block 范围和权重的专家下标
* 任务队列自己定义为显存上的 int32 `[num_tasks, 4]`，每行 `[expert_idx, m_start, m_size, ready]`
* 一次 launch 领一个任务，`task_idx` 由 launch 参数给出：流内天然串行，所以第 i 个 kernel 就是领第 i 个到达的 chunk，CPU 不参与调度
* swiglu 新写了一个 persistent kernel（`o2 = silu(o1[:, :N]) * o1[:, N:] * prob`，fp32 计算），同样从队列领任务
* 新 API 为 `deep_gemm.bf16_chunk_gemm_nn` / `deep_gemm.chunk_weighted_swiglu`，原 `bf16_gemm_nn` 用法不变
* 沿用 m-grouped 的 layout 启发式，所以自动拿到 swap-AB + 2-CTA；要求 `m_start` 按 BLOCK_M(=128) 对齐

8.25: 加上 `ready` 的 spin-wait，单卡用 CPU 写 pinned memory 模拟 chunk 到达
* kernel 开头每个 CTA 由 0 号线程 `ld.acquire.sys` 轮询 `ready`，然后把 task 描述符 stage 到 shared memory 再广播给全 CTA；带 30e9 cycle(约15s) 的超时保护，避免死锁挂住整卡
* 用 system scope 而非 gpu scope，所以队列放显存（真实 DeepEP）还是放 pinned host memory（单卡模拟）都成立——CPU 一句普通 store 就够，不需要任何 CUDA API，也不会像 paddle 的 fill 算子那样阻塞
* 坑：一开始让全部 256 线程直接读队列，pinned memory 的 uncached 读无法合并，单 kernel 多花约 2ms（108 个 kernel 从 7ms 涨到 239ms）；改成单线程读 + shared memory 广播后降到约 0.3ms/kernel
* 剩下这 0.3ms/kernel 是 96 个 CTA 各自读一次 host memory 的 PCIe 延迟，只存在于单卡模拟里
* 加了显存队列版本（`test_chunk_spin_gpu.py`）：producer 换成显存上写信号的 kernel，占 1 个 SM（整卡 148 SM，计算占 96，还剩 52 给通信和 zip），spin 开销直接落进噪声，说明最终形态没有这个问题；因此 `.sys` scope 保留不变（顺带兼容 CPU 写信号的调试路径），不需要退化成 `.gpu` scope

8.25: 补上 down 之后的 token 完成信号
* 新增 `deep_gemm.chunk_signal_token_done(row_to_token, token_done, task_queue, task_idx)`：在每个 chunk 的 down 之后紧跟发一个 persistent kernel，对该 chunk 的每一行做 `red.release.sys.global.add.s32`，把 `token_done[token_idx]` 加一
* 粒度按 DESIGN 取 chunk 级（"每组算子完成任务后"），所以做成独立 kernel 而不是塞进 down 的 epilogue：epilogue 里一行要等它全部 n_block 都存完才算完成，得引入跨 CTA 计数；独立 kernel 靠 kernel 边界 + PDL 就天然保证 o3 已可见，代价只有几 µs。想把信号提前到 m_block 粒度的话再考虑前者
* 需要 `row_to_token[num_unzipped_tokens]`（unzip 排布的行 → 不重复 token 下标，padding 行为 -1），这个映射得由 unzip 那边给出；`token_done[num_recv_tokens]` 每轮开始要清零
* 用 release 语义是为了让 zip 看到计数时一定能看到 o3 的数据
* 顺手修了个坑：swiglu 原来只写 m_size 行，导致 down 的最后一个 block 会读到 o2 的 padding 行（上一轮的残留/NaN）。虽然结果只落在 o3 的 padding 行上无害，但会让 NaN 检查失效；现在 swiglu 把 padding 尾巴补零，整条链路无 NaN
* TODO：性能测量（chunk 流式 vs baseline 串行），以及和通信侧真正对接

8.25: 把四个 chunk 测试合并成 `test_chunk.py`，用 `--arrival {ready,cpu,gpu}` 选到达方式，down signal 默认都发、`--check-signal` 控制是否逐 chunk 校验
* 顺手修了个坑：producer kernel 原来用连续多次 `__nanosleep` 来等间隔，实测严重超时（请求 82ms 的到达延迟，total 跑到 146ms）；改成用 `clock64()` 计圈（`wait_cycles`），host 侧用 `cudaDeviceGetAttribute(cudaDevAttrClockRate)` 把 ns 换成 cycle，现在 total 98.7ms ≈ 82ms + 计算尾巴
* CUDA 13 删了 `cudaDeviceProp::clockRate`，只能走 `cudaDeviceGetAttribute`


