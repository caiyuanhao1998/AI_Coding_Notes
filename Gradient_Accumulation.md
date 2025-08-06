### 1. 极简使用 Gradient Accumulation

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

pending = 0
buf = []  # 只为知道尾组大小，也可用计数器

for x, y in loader:              # loader 的 batch_size = B
    buf.append(1)
    group_size = accum_steps if len(buf) < accum_steps else len(buf)

    # criterion 是损失函数, model(x) 是模型的预测, y 是 label
    # group_size 表示总的 batch size, 总共有那么多数量的 sample，每一个 Loss 本身不要求 mean
    # 这个 backward() 对当前前向图做反向传播，计算每个可训练参数的 ∂loss/∂𝜃，并累加到 param.grad 里

    (criterion(model(x), y) / group_size).backward()

    '''
        [1] 每次 loss.backward() 完成后，这次前向的计算图（activations 等）会被释放（默认 retain_graph=False）。
        下一次 micro-batch 会重新前向、重新构图，因此激活显存不会随着累积步数线性增长

        [2] param.grad 张量会在第一次 backward() 时分配，此后在组内复用并累加（显存基本恒定）

        [3] 只有在你 zero_grad(set_to_none=True) 时，才把 grad 设回 None，从而释放梯度缓冲；否则若置零，不会释放那块内存
    '''

    pending += 1

    
    if pending == accum_steps:
        # 用当前 param.grad 对参数做一次更新（如 SGD/Adam），然后 param.data 被改变
        optimizer.step()
        # 把 param.grad 清空 (置 0 或置 None)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        buf.clear()

# 处理最后不足一组
if pending:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```


### 2. 不使用 Gradient Accumulation，直接放大 batchsize

```python
big_loader = DataLoader(dataset, batch_size=B*4, shuffle=True, drop_last=False)

optimizer.zero_grad(set_to_none=True)
for x, y in big_loader:
    loss = criterion(model(x), y)   # 不需要除以 4
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```
