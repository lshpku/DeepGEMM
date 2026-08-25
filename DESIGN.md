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
  * 由于 n_chunks 在计算开始前已知，所以在开始时 CPU 直接往计算流发射 n_chunks 组 (gateup, swiglu, down) 算子（也就是一共 n_chunks*3 个算子）
  * 每个算子的输入指针不指定，靠从队列里获取，由于计算流内部天然阻塞，所以不存在一个算子抢另一个算子的任务的情况，任务一定是一个一个完成，最后恰好 n_chunks 组算子做完所有任务
  * 每组算子完成任务后，往一个表记录每个 token 的完成情况（已经被几个专家完成）

4. zip 与 combine 部分，zip 是一个单独的 persistent kernel，combine 也使用 DeepEP，增加信号等待逻辑：
  * zip 通过读计算完成队列，对于一个不重复 token，当它 topk 的所有专家都计算完就对其进行求和操作，写到 combine 的输入 buffer
  * combine 和 dispatch 同一个流，所以一定在 dispatch 完成后才启动
  * comine 同样增加一个等待逻辑，等输入 buffer 里的一个 token 就绪之后才发出
  * zip+comine 逻辑同样由另一组同事开发，已经论证过正确性，我这里只需要正确给出 token 完成信号


<b>关于chunk的说明：</b>其实通信并不是严格按chunk发送的，它相当于是细水长流地并发收到各个专家的token，然后push到各个专家的buffer上，我所谓的“概念上”指的是当一个专家每凑够chunk数量的token时，就理解为它的一个chunk到达了，并不是说一个chunk突然就一次性到达了。使用chunk这个概念是为了保证GEMM的连续性，因为我们是训练场景，如果每到一个token就计算，性能肯定非常差。


## 计算部分设计细节

下面的细节可以讨论

### kernel选型

目前 gateup 和 down 想用 deep_gemm.bf16_gemm_nn 这个接口对应的底层算子（我没仔细看代码，但是实测性能很好，对于 chunk=4096、16个专家、平均每个专家8192个 token 的场景，它只比调用单个 group_gemm 慢了3%）

我知道 deepgemm 是支持动态M的，它可以在领任务的时候才知道M是多少，NK肯定是launch时已知的；当然我不清楚动态M对性能影响多大，需要根据测试决定是用动态M还是总是向chunk对齐允许浪费

swiglu 就新实现一个就行，我们实际用的是 weighted_swiglu，就是 router_score 在这里乘进去，paddle 原有算子不支持动态 M，所以需要新写一个

计算kernel都需要使用persistent的形式，就是使用固定数量的SM，SM自己分配任务；目前安排的是计算流（gateup+swiglu+down）用 96SM，其他是给通信和zip用的；大家统一用2-CTA形式，这样launch不会导致1-CTA和2-CTA之间冲突，其实官方DeepEP早就默认是2-CTA了，反而计算这边很多kernel迟迟没跟进


### buffer设计

由于很多变量命名比较混乱，先去歧义一些定义：
`seq_len`：dispatch 前每个 rank 的 token 数量，也就是写在训练 recipe 里那个值
`num_recv_tokens`：dispatch 后当前 rank 收到的**不重复 token** 的数量，每轮 microbatch 都不同
`num_experts`：每个 rank 上的专家数（即本地专家数）
`tokens_per_expert[num_experts]`：dispatch 后当前 rank 上每个 expert 收到的 token 数量，是一个数组
`num_unzipped_tokens`：sum((n + 127) // 128 * 128 for n in tokens_per_expert)，也就是向 128 对齐后的展开的总 token 数，向 128 对齐是 GEMM 的固有要求

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

目前通信和计算之间的信号 buffer 的设计还没定，通信那边也都未定稿，我们开发时自己先定义一套就行，我们这边保证我们的 acquire/release 语义是正确的就行

和原来串行方式对比的话，相当于原来是 unzipped_tokens 已经完全就绪才开始算 gateup，现在是 unzipped_tokens 里面随机的某个专家的 chunk 先到，哪个专家先到不确定，但是一个专家内一定是 chunk0->chunk1->chunk2 这样的顺序，这个通信那边已经保证了

我们计算要做的就是当一个 chunk 就绪时，立即做它的 gateup+swiglu+down，然后更新输出的完成信号给后面的 kernel 用

由于 unzipped_tokens->o1->o2->o3 都是一样的布局，输入是第几个 token 输出就写到第几个 token 的位置，无需任何转换
