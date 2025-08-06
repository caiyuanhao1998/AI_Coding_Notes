### 1. 极简使用 Gradient Accumulation

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

pending = 0
buf = []  # 只为知道尾组大小，也可用计数器

for x, y in loader:              # loader 的 batch_size = B
    buf.append(1)
    group_size = accum_steps if len(buf) < accum_steps else len(buf)

    (criterion(model(x), y) / group_size).backward()
    pending += 1

    # 到组末（凑满4个）就更新
    if pending == accum_steps:
        optimizer.step()
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
