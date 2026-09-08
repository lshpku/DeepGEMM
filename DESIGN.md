# 基于跨Stream信号的细粒度计算-通信Overlap实现（GEMM部分）

## 项目背景

这是我们 Paddle 预训练团队的一个探索性研究项目，本人负责 GEMM 部分的实现。

当前我们在 EP 场景下的 MoE 执行方法如下：
1. 调用 DeepEP dispatch 进行 all2all 通信
2. 调用 unzip 算子将收到的 token 分发到各个专家的输入 buffer
3. 调用 DeepGEMM 的 gemm/group_gemm（当token数量大时普通gemm更快）计算专家的 FFN
4. 调用 zip 算子聚合（求和）各个专家的输出 buffer，得到无重复的 tokens
5. 调用 DeepEP combine 进行 all2all 通信

上述方案作为 baseline 性能尚可，但在通信时间较长的场景下，由于计算和通信无法 overlap，导致计算或通信带宽的浪费。

现在我们提出一个新方案，就是通过多流同时 launch 计算和通信，并通过跨 Stream 信号进行同步，让计算和通信能有效 Overlap 起来，该方案简述如下：

1. 进入 MoE 模块后，CPU 同时 launch 计算流和通信流的算子

2. dispatch 与 unzip 部分，仍然使用 DeepEP，但增加一个 chunk 的概念：
  * 通过融合 dispatch+unzip，让抽象层面上收到的 token 可以看成按专家分 chunk 抵达
  * 可以理解为比如 expert1 的 chunk0 先抵达，然后 expert3 的 chunk0 抵达，然后 expert2 的 chunk0 抵达，……，然后又轮到 expert1 的 chunk1 抵达，……，直到结束
  * 收到一个 chunk 意味着有了属于某个专家的 chunk 个连续的 token 可以立即用于计算
  * 需要注意的是哪个专家的哪个 chunk 先抵达是运行时随机的，但是总 chunk 数是知道的，DeepEP 在通信开始前已完成路由信息交换
  * chunk 大小是固定的比如 4096，但是对于一个专家最后的余数部分是可能不满 4096 的，计算 kernel 应当能够处理这种情况且避免计算空转
  * 目前已由其他组同事完成 chunk 到达信号记录的实现，也就是显存上有一个队列，记录当前收到的 (expert_id, chunk_id) 序列，但 kernel 还在调试还没给到我，开发时可以自己写一个队列在单卡模拟这种情况，这样也方便单卡调试

3. 计算部分，使用 DeepGEMM 的 bf16 矩阵乘 + 自己实现的 swiglu，具体调用方法为：
  * 由于 n_chunks 在计算开始前已知，所以在开始时 CPU 直接往计算流发射 n_chunks 组 (gateup, swiglu, down, done) 算子（也就是一共 n_chunks\*4 个算子，融合 gateup+swiglu 的话就是 n_chunks*3）
  * 每个算子的输入指针不指定，靠从队列里获取，由于计算流内部天然阻塞，所以不存在一个算子抢另一个算子的任务的情况，任务一定是一个一个完成，最后恰好 n_chunks 组算子做完所有任务
  * 每组算子完成任务后，首先往一个表记录每个 token 的完成情况（已经被几个专家完成），对于已经被所有专家完成的 token 即可发往下一步；由于这个 “记录” 操作和 gemm 不好融合，所以新增一个 done 算子来进行记录和入队操作

4. zip 与 combine 部分，zip 是一个单独的 persistent kernel，combine 也使用 DeepEP，增加信号等待逻辑：
  * zip 通过读计算完成队列，对于一个不重复 token，当它 topk 的所有专家都计算完就对其进行求和操作，写到 combine 的输入 buffer
  * combine 和 dispatch 同一个流，所以一定在 dispatch 完成后才启动
  * comine 同样增加一个等待逻辑，等输入 buffer 里的一个 token 就绪之后才发出
  * zip+comine 逻辑同样由另一组同事开发，已经论证过正确性，我这里只需要正确给出 token 完成信号


<b>关于chunk的说明：</b>其实通信并不是严格按chunk发送的，它相当于是细水长流地并发收到各个专家的token，然后push到各个专家的buffer上，我所谓的“概念上”指的是当一个专家每凑够chunk数量的token时，就理解为它的一个chunk到达了，并不是说一个chunk突然就一次性到达了。使用chunk这个概念是为了保证GEMM的连续性，因为我们是训练场景，如果每到一个token就计算，性能肯定非常差。

<b>关于zip：</b>目前计算这边自己也实现了一个 zip，融合到 signal_done 里了，每个 chunk 调用一次，用于替代通信的 persistent zip，理论上可以给计算换更多的 SM；目前实测我们的 zip 性能更好，之后基本只维护我们的 zip 即可。


## 计算部分设计细节

下面的细节可以讨论

### kernel选型

目前 gateup 和 down 想用 deep_gemm.bf16_gemm_nn 这个接口对应的底层算子（我没仔细看代码，但是实测性能很好，对于 chunk=4096、16个专家、平均每个专家8192个 token 的场景，它只比调用单个 group_gemm 慢了3%）

deep_gemm 是支持动态 M 的，它可以在领任务的时候才知道 M 是多少，NK 则是 launch 时已知的；动态 M 对性能影响很小，我们用动态 M 可以让专家的余数部分避免冗余计算

需要实现一个新的 swiglu 或直接将 swiglu 融合到 gateup 的 epilogue；我们的模型算法实际用的是 weighted_swiglu，就是 router_score 在这里乘进去而不是在 zip 的时候；由于 Paddle 原有算子不支持动态 M，需要新写一个

计算 kernel 都需要使用 persistent 的形式，就是使用固定数量的 SM，SM 自己分配任务；目前安排的是计算流（gateup+swiglu+down+done）用 96SM，其他是给通信和 zip 用的；大家统一用 2-CTA 形式，这样 launch 不会导致 1-CTA 和 2-CTA 之间冲突，其实官方 DeepEP 早就默认是 2-CTA 了，反而计算这边很多 kernel 迟迟没跟进


### buffer设计

由于很多变量命名比较混乱，先去歧义一些定义：
`seq_len`：dispatch 前每个 rank 的 token 数量，也就是写在训练 recipe 里那个值
`num_recv_tokens`：dispatch 后当前 rank 收到的**不重复 token** 的数量，每轮 microbatch 都不同
`num_experts`：每个 rank 上的专家数（即本地专家数）
`tokens_per_expert[num_experts]`：dispatch 后当前 rank 上每个 expert 收到的 token 数量，是一个数组
`num_unzipped_tokens`：sum((n + 127) // 128 * 128 for n in tokens_per_expert)，也就是向 128 对齐后的展开的总 token 数，向 128 对齐是 GEMM 的固有要求

对 token 在 buffer 中的顺序消歧义如下：
`sequence序`：dispatch 之前的 token 顺序，也就是 hidden_states 中的顺序，确定性
`DeepEP序`：原版 DeepEP dispatch 收到不重复 token 的顺序，combine 的输入也用该顺序，确定性
`前向atomic序`：前向 dispatch + unzip 后得到的顺序，由于使用了 atomic 先到先放，非确定
`反向atomic序`：反向 dispatch + unzip 后得到的顺序，同样非确定，且和前向的顺序不一样，因此在一次前反向中其实存在两种不同的随机顺序

GEMM 的输入这边仍然沿用原来的 unzip 输出的 buffer 设计，里面每个专家的 token 连续排列，即前 tokens_per_expert[0] 个 token 是专家0的，**向128对齐后**，接下来的 tokens_per_expert[1] 个 token 是专家1的，以此类推
* `unzipped_tokens`[num_unzipped_tokens, hidden_size] bf16
* `unzipped_probs`[num_unzipped_tokens] fp32

权重则是两个融合矩阵
* `w_gateup`[num_experts, hidden_size, 2*intermediate_size] bf16
* `w_down`[num_experts, intermediate_size, hidden_size] bf16

计算过程中的中间变量定义如下：
* `o1`[num_unzipped_tokens, 2*intermediate_size] bf16：gateup的输出
* `o2`[num_unzipped_tokens, intermediate_size] bf16：weighted_swiglu的输出
* `o3`[num_unzipped_tokens, hidden_size] bf16：down的输出


### 信号设计

目前通信和计算之间的信号 buffer 的设计还在迭代中，只定义了当前开发阶段所需的，如果发现有需要可以继续新增
我们需要保证我们这边保证我们的 acquire/release 语义是正确的，通信那边也会保证正确的语义

dispatch 给到计算的除了 unzipped_tokens 和 unzipped_probs，还有一个任务队列：
`task_queue`[num_chunks, 4] int32 : 记录每个 chunk 的描述符和就绪信号，格式为 [expert_idx, m_start, m_size, ready]
* num_chunks 是可以提前算出来的，因为 dispatch 开始前就已经知道 tokens_per_expert，则 num_chunks = sum(ceil(n / chunk) for n in tokens_per_expert)
* expert_idx 是该 chunk 属于哪个本地专家
* m_start 是该 chunk 在 unzipped_tokens 里的偏移 token 数
* m_size 是 chunk 包含的 token 数，一般为 chunk 大小，但对于一个专家的余数部分是可以小于 chunk 大小的，计算这边用了动态 M 的 kernel
* 该队列是一个 FIFO 队列，ready 的任务会按顺序连续 push 进来，计算这边也使用递增方式按顺序等待 ready 信号即可

dispatch 的一个输出可以用于专家下标排序：
`recv_token_indices`[num_recv_tokens, topk] int64 : 使用 DeepEP 序，收到的 token 属于本地哪些专家（最少1个，最多topk个，使用本地专家下标，无效部分用-1填充，行内未排序）

dispatch 给了两张映射表用于前向 atomic 序和 DeepEP 序之间的转换，该表随着 receiver 更新，收到 token 后才赋值；当然通信算子保证我们读到 ready 时这个 chunk 内所有 token 的信息都就绪了：
`atomic_to_zip`[num_unzipped_tokens] int32 : 使用 atomic 序，记录一个 token 指向 DeepEP 序里的哪个下标，padding 位置填 -1
`zip_to_atomic`[num_recv_tokens, topk] int32 : 使用 DeepEP 序，位置和 recv_token_indices 一一对应，记录对于每个有效的 token，zip 的时候应该去 o3 的哪个下标读，无效位置填 -1

为了让 zip 算子知道每个 token 的有效 topk 数，dispatch 还会给一张计数表，同样是随 ready 动态更新的：
`num_valid_topk`[num_recv_tokens] int32 : 使用 DeepEP 序，记录每个 token 在本地有几个专家，其值等价于**通信完成后** recv_token_indices 里面每行非 -1 的数量，但是在运行时不相等，因为 recv_token_indices 的一行是分专家更新的，一个 chunk 就绪时只保证这个 chunk 所属专家在 recv_token_indices 里面的槽位就绪，不保证这一行所有专家都就绪，导致数少了；num_valid_topk 则是通过冗余更新解决这个问题，一个 token 的每个专家的 chunk 发布时都会重新写一次 topk 值

done 算子需要维护以下两张表用于记录 token 完成情况：
`token_done`[num_recv_tokens] int32 : 使用 DeepEP 序，当一个 token 的计数等于 num_valid_topk 里的对应值时，说明它被所有专家计算完了，就可以进行 zip 了；这张表只在计算内部用，不需要给通信
`zip_task_queue`[num_recv_tokens] int32 : 是一个连续填充的队列，内容为计算完的 token 在 DeepEP 序中的下标，完成的 token 会被依次 push 进来；该队列初始值为 -1，这样当读到非 -1 的值时就知道就绪了，不需要额外 ready 信号；该队列会给到通信

zip 和 combine 之间通过一个表记录每个 token 是否可以被 combine：
`zip_done`[num_recv_tokens] int32 : 使用 DeepEP 序，仅使用 0/1 值表示，由 zip 向已经完成 zip 的 token 位置写 1


### 反向Kernel

先梳理一下计算部分的前向 kernel：
* gateup: 调用 bf16_chunk_gemm_nn (x -> o1)
* swiglu: 调用 chunk_weighted_swiglu 或与 gateup 融合 (o1,probs -> o2)
* down: 调用 bf16_chunk_gemm_nn (o2 -> o3)
* zip: 调用 chunk_zip (o3 -> out)

那么根据反向的公式，反向所需的 kernel 为：
* zip_grad: 不需要，由通信提供；zip_grad 本质上就是 unzip，只是数据搬运，无计算，unzip 已经融合在通信 kernel 里
* down_grad: 调用 bf16_chunk_gemm_nt (do3 -> do2)
* swiglu_grad: 这个最复杂，需要输入前向激活 o1 和 probs，重计算 o2' 给 wgrad 用，同时根据 do2 算出 do1，以及计算 dprobs_recv；由于过于复杂，目前不和 down_grad 做融合 (o1,probs,do2 -> o2',do1,dprobs_recv)
* gateup_grad: 调用 bf16_chunk_gemm_nt (do1 -> dx)
* unzip_grad: 调用 chunk_zip，累加方式和前向相同 (dx -> drecv_x)

关于 nn/nt：两者底层实现相同，是同一个 kernel 模板 kMajorB 的两个实例化（nn 是 MN-major B，nt 是 K-major B，后者其实才是库里的原生形态）；信号等待和 chunk 调度逻辑完全共用，零转置零拷贝；nt 与 nn 在相同 (M,N,K) 下性能持平，见 tests_overlap/test_chunk_nt.py

wgrad 不需要我们做，我们只要把 (x',do1)、(o2',do3) 给出即可，用户会调用专门的 k_group_gemm 算子来计算，wgrad 不在关键路径上；注意 padding 部分要置 0，do1 和 do3 可能需要专门后处理置 0，x' 和 o2' 作为新的输出也要专门处理

drecv_probs 是一个和 recv_token_indices 相同 shape 和 token 槽位的 tensor，算出每个 token 的 dprob 后直接 scatter 到 drecv_probs 中，后续无累加操作

但是这里有个很重要的顺序问题，就是前向 atomic 序和反向 atomic 序是不同的
* 对于主干计算 (dout->do2、do1->dx、dx->drecv_x) 不用管，因为这些用的都是反向 atomic 序，token 位置能够对齐，unzip_grad (实际是 zip) 也能够恢复顺序
* 但是对于 swiglu_grad 不是，因为它的输入 o1 和 probs 是前向 atomic 序，读的时候需要先转换成反向 atomic 序，它输出的 o2 也应该是反向 atomic 序
* 另外，前向没有保存 x 激活，保存的是 recv_x (也就是没 unzip 的，使用 DeepEP 序)，我们可能需要写一个 wgrad 专用的 unzip 来解压出反向 atomic 序的 x'，不过这是后话了，先把主干梯度做出来

注：DeepEP 用的是前反完全对称的实现（或者说它就没有反向的概念），combine 的反向就是一模一样的 dispatch，它在给我们反向 atomic 序的 do3（也就是 unzipped_tokens）的同时也会给出反向 atomic 序的 atomic_to_zip/zip_to_atomic；但是 recv_token_indices 是前反向相同的，包括 token 槽位也相同

反向 atomic 序到前向 atomic 序转换的一种做法为：通过反向 atomic_to_zip 拿到一个 token 在 DeepEP 序中的位置 -> 通过 recv_token_indices 获取该 token 所属专家的槽位 -> 在前向 zip_to_atomic 的对应槽位中拿到该 token 在前向 atomic 序中的下标


### 离线顺序转换函数

需要一些顺序转换函数来高效压测正确性，用 sort 固然可以，但是非常影响大规模训练下的压测效率
以下函数都是离线函数，在通信全部结束后才进行，不需要处理任何一致性逻辑

`sort_unzip_map`
* 输入：前向的 zip_to_atomic 等变量
* 输出：
  * ordered_to_zip [num_unzipped_tokens] int32 : 和前向 atomic_to_zip 的 shape 一样，但是每个专家内的 token 是按照它们在 DeepEP 序中的顺序排序的，对应 paddle 标准的 unzip 的行为，这样用户就可以通过一次 gather 从前向未 unzip 的 recv_x [num_recv_tokens, hidden] 中解压出 paddle 标准顺序的 unzipped_tokens；padding 位填 -1

`sort_atomic_map`
* 输入：反向的 zip_to_atomic 等变量
* 输出：
  * ordered_to_atomic [num_unzipped_tokens] int32 : 将标准 unzip 序映射到反向 atomic 序的映射表，这样用户可以通过一次 gather 从反向 atomic 序的 do1/o2_bwd/do3 得到 paddle 标准顺序的 do1/o2_bwd/do3；padding 位填 -1

`token_gather`
* 类似 paddle.gather(x, index, axis=0)，但是对于 -1 的下标直接写 0（paddle 对于 -1 下标是理解为 len(x)-1，这不符合 padding 的要求）
