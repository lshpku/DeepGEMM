import paddle


def make_deepep_layout(ep_size, seqlen, local_e, topk, alignment=128, router_bias=0.1):
    """
    模拟在一个 EP 组内, 每个 rank 有 E 个专家的情况下, rank 0 收到的 dispatch+unzip 后的
    (recv_token_indices, tokens_per_expert).
    """
    # 模拟全局 token 打分
    scores = paddle.randn([ep_size * seqlen, ep_size * local_e])
    if router_bias:  # add some system bias to experts
        scores += paddle.randn([ep_size * local_e]) * router_bias
    _, topk_indices = scores.topk(topk)

    # 只保留命中 rank0 的 token, 故 topk_indices 里每一行至少有一个非 -1 值
    topk_hit = topk_indices < local_e
    token_hit = topk_hit.any(axis=1).nonzero().squeeze(1)
    topk_indices[~topk_hit] = -1
    topk_indices = topk_indices[token_hit]

    # DeepEP 并未对每行内容进行排序, 甚至有效值和 -1 是交错排列的, 这里对每行进行乱序
    row_perm = paddle.randn(topk_indices.shape).argsort(axis=1)
    topk_indices = topk_indices.index_sample(row_perm)

    tokens_per_expert = paddle.sum(
        paddle.arange(local_e)[:, None] == topk_indices.flatten(), axis=1).tolist()

    m_start, m_indices = [0], []

    for expert_idx, n in enumerate(tokens_per_expert):
        n_aligned = (n + alignment - 1) // alignment * alignment
        m_start.append(m_start[-1] + n_aligned)
        m_indices.append(paddle.full([n_aligned], expert_idx, dtype="int32"))

    m_indices = paddle.concat(m_indices)

    return topk_indices, tokens_per_expert, m_start, m_indices


def make_atomic_layout(topk_indices, tokens_per_expert, m_start):
    """
    根据 DeepEP 序构造一份随机顺序的 atomic 序.
    DeepEP 序前反向是相同的, 但是 atomic 序不同, 可以调用两次本函数模拟不同的 atomic 序.
    """
    atomic_to_zip = paddle.full([m_start[-1]], -1, dtype="int32")
    zip_to_atomic = paddle.full(topk_indices.shape, -1, dtype="int32")

    for expert_idx, (n, offset) in enumerate(zip(tokens_per_expert, m_start)):
        # 选出属于 expert_idx 的 token 并打乱顺序
        slot_hit = topk_indices == expert_idx
        token_idxs = slot_hit.any(axis=1).nonzero().squeeze(1).cast("int32")
        assert len(token_idxs) == n
        perm = paddle.randperm(n)

        atomic_to_zip[offset : offset + n] = token_idxs[perm]

        zip_to_atomic[slot_hit] = paddle.empty([n], dtype="int32").scatter_(
            perm, paddle.arange(n, dtype="int32") + offset
        )

    return atomic_to_zip, zip_to_atomic
