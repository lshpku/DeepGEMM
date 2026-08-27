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
python tests_overlap/test_fused_swiglu.py
```

`test_gemm_baseline.py`：对比group_gemm和chunk的性能测试，一个是调用单次group_gemm，一个是分chunk调用，实测性能差距很小，chunk方案仅慢2%，说明分chunk几乎不影响性能

`test_chunk.py`：chunk 流式全链路（gateup, swiglu, down, signal）的正确性测试。按真实路由构造布局（默认 16384 个不重复 token，topk=8，每专家区域向 128 对齐，所以有真的 padding 行），先在异步流上把全部 kernel 发出去让它们卡在 spin-wait 上，最后与 group_gemm 逐位比对 o1/o2/o3（只比真实行）并检查 `token_done` 每个 token 恰好被记 topk 次。

`test_fused_swiglu.py`：单算子测试，验证把 weighted SwiGLU 融进 gateup epilogue 的正确性（o1/o2 都与两 kernel 路径逐位一致）和代价（融合只多 2us，独立 swiglu kernel 要 26us）


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

8.25: 修掉 chunk gemm 比原版慢 45% 的问题：领任务的超时保护里那句 `trap` 拖垮了整个 kernel
* 现象：同样 41 个 chunk 的 gateup，`bf16_gemm_nn` 分块调用 3.6ms、`m_grouped` 分块调用 3.8ms，而 `bf16_chunk_gemm_nn` 要 8.4ms；单 kernel 隔离测量 167us vs 114us
* 排查过程：先确认不是 layout 配置（chunk 走的是 m-grouped 那套强制 layout，模板参数和 `m_grouped` 的 cubin 完全一致，只有 GemmType 不同），也不是大 M 的 tensor map 或 `m_start` 偏移（把 chunk kernel 作用在 4096 行的切片上、offset 换成 0，都一样慢）；`cuobjdump -res-usage` 显示 chunk 版多了 `STACK:24`
* 根因：`wait_task_ready` 的超时保护写在 `if (threadIdx.x == 0)` 这个 divergent 区里，只要里面有可达的 `trap`（`DG_DEVICE_ASSERT` 更糟，它还调 `printf`，会给整个 kernel 一个 local memory frame），ptxas 就会对整个 GEMM 降级，稳定损失约 45%
* 改法：`wait_task_ready` 只返回是否超时，把 `timed_out` 连同任务描述符一起 stage 到 shared memory，`__syncthreads()` 之后全 CTA 统一 `DG_TRAP_ONLY_DEVICE_ASSERT`；这样超时保护还在，但 trap 在 uniform 区里，代价为零
* 结果：单 kernel 回到 114us（和 `m_grouped` 持平），41 chunk 总时间 4.0ms vs `m_grouped` 3.8ms；o1/o2/o3 和 `token_done` 仍然全部逐位一致

8.25: 把 weighted SwiGLU 融进 gateup 的 epilogue，chunk=4096 时 swiglu 从 26us 降到 2us
* DeepGEMM 原生只有 `epilogue_type_t::apply_index_n` 这一个 hook（`EpilogueHeadSplits` 在用），只能改 TMA store 的 N 下标，不能改数值，所以融合得自己加一段
* 关键观察：swap-AB 的 epilogue 每个 store stage 会把 `[16, BLOCK_N=128]` 的结果按 128B swizzle 落到 shared memory，而这块 smem 正好是两个 `[16, 64]` 的 atom；只要权重列按 64 一组交错成 `[gate0:64, up0:64, gate64:128, ...]`（运行时零成本），gate 和 up 就落在同一个 stage 的两个 atom 里，直接在 epilogue 里读回来算完就行
* 实现：`sm100_store_cd_swap_ab` 加 `kWithFusedSwiGLU` 模板参数，128 个 epilogue 线程每人从 smem 取一对 16B（gate/up 各 8 个 bf16），fp32 算完直接 16B 写 o2；o1 的 TMA store 完全不变（反向要用），o2 因此不再需要整块常驻显存，只要一个 chunk 大小的 buffer
* API 就挂在 `bf16_chunk_gemm_nn` 上：多给 `o2`/`probs` 两个可选参数就启用融合，不另开接口，也不做非 task queue 的版本（baseline 用旧的 `bf16_gemm_nn` 就够）
* 精度：读的是已经 cast 成 bf16 的 o1，和两 kernel 路径完全一样，实测 o1/o2 都与 paddle 参考逐位一致
* 大坑：一开始用 `g / (1.0f + __expf(-g))`，融合开销 26us；换成 `__fdividef` 后只剩 2.1us。IEEE 除法在 epilogue 这种和 MMA 抢线程的地方极贵，独立 swiglu kernel 同样吃这个亏（38us → 26us）
* 现状（m=4096, H=4096, I=2048, 96 SM）：纯 gateup 111us，融合版 113.6us，独立 swiglu 还要 26us；独立 kernel 只跑到 1.96 TB/s，仍然是延迟受限（每线程 in-flight 太少），但既然融合几乎免费就不再优化它
* 下一步：把融合接到 chunk 版 GEMM（`MGroupedChunk`）上，o2 换成 chunk 大小的复用 buffer，并把交错权重的准备放到上层

8.27: done 算子扩充成完整的完成信号发布，直接给出 zip 要的 `zip_task_queue`
* API 变为 `chunk_signal_token_done(atomic_to_zip, num_valid_topk, token_done, zip_task_queue, zip_queue_tail, task_queue, task_idx)`，原 `row_to_token` 按 DESIGN 统一改名 `atomic_to_zip`
* 判定"我是最后一个专家"必须用带返回值的原子加（新增 `ptx::atomic_add(int*)`，`atom.gpu.global.add.s32`），按 `old + 1 == num_valid_topk[token]` 判断：只有一个线程能看到相等，所以每个 token 恰好 push 一次。原来的 `red.release` 拿不到返回值，用不了
* 队列不需要额外的 ready 列：payload 只有一个 int，单次 4B store 天然原子，队列初始化为 -1，读到非 -1 即就绪。抢槽位用一个 int32 `zip_queue_tail` 做 `atomicAdd`，是纯计算侧的内部状态，不给通信
* 但单靠一个 tail 指针不行：抢槽位有序而写 payload 无序，zip 看到 tail 前移时对应的 slot 可能还没落盘，所以"值自己当 ready"这一层是必须的
* 内存序只在 push 那一次 store 用 `.release.sys`：所有 chunk 都在同一条计算流上串行，一个 token 其余专家的 o3 早在本 kernel 启动前就写完了，所以计数器本身用 relaxed 就够，不需要用 acq_rel 去接别人的 release
* `num_valid_topk` / `atomic_to_zip` 用普通 load 读：CTA 里 0 号线程对 `ready` 的 `ld.acquire` 已经给全 CTA 建立了顺序，且这个 chunk 的表项之后不再变（`num_valid_topk` 是冗余重写同一个最终值，幂等）
* 测试侧：三张信号表每轮都要重置（`token_done` 清 0、`zip_task_queue` 清 -1、`tail` 清 0），性能循环里重复跑同一批 chunk 会把计数叠加到超过 `num_valid_topk` 而触发 assert
* 结果：`--check-signal` 通过（39 chunk / 109707 token，逐 chunk 校验队列内容和 `token_done` 完全一致，最终队列恰好是全部 token 的一个排列），`--arrival ready/cpu` 通过
* 遗留：`--arrival gpu` 在 launch producer 时报 719（unspecified launch failure）。确认与本次改动无关——把 done 算子整个去掉、或关掉 swiglu 融合都一样失败，而 producer kernel 单独跑正常，怀疑和计算 kernel 全部卡在 spin-wait 时再往另一条流 launch 有关，待查


