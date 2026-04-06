# 启用Epilog - 保存对话记录

## 概述

修改后的代码现在支持自动保存对话记录（Episode Log）到数据库。以下是如何使用这个功能的指南。

## 快速开始

### 1. 修改 `minimalist_demo.py`

已经配置好了两个关键参数：

```python
asyncio.run(
    run_async_server(
        model_dict={...},
        sampler=UniformSampler(),
        push_to_db=True,      # 启用数据库保存
        tag="qwen_test",      # 用于组织和查询的标签
    )
)
```

**参数说明：**
- `push_to_db=True`：启用对话记录保存
- `tag="qwen_test"`：为本次运行的所有对话添加标签，方便后续查询

### 2. 运行对话

```bash
python examples/minimalist_demo.py
```

对话完成后，结果会自动保存到数据库。

## 查看保存的对话

### 方法1：使用查询脚本（推荐）

已经创建了 `view_episodes.py` 脚本来查看保存的对话：

#### 查看所有标签为 "qwen_test" 的对话：
```bash
python examples/view_episodes.py list qwen_test
```

#### 查看特定对话的详细内容：
```bash
python examples/view_episodes.py show <episode_id>
```

#### 导出对话为JSON文件：
```bash
python examples/view_episodes.py export qwen_test episodes.json
```

### 方法2：在Python中直接查询

```python
from sotopia.database import EpisodeLog

# 查询所有 tag=qwen_test 的对话
episodes = EpisodeLog.find(EpisodeLog.tag == "qwen_test").all()

# 遍历和查看对话
for episode in episodes:
    print(f"Episode ID: {episode.pk}")
    print(f"Agents: {episode.agents}")
    print(f"Messages: {episode.messages}")
    print(f"Reasoning: {episode.reasoning}")
    print(f"Rewards: {episode.rewards}")
```

## 对话数据结构

保存的每个对话（EpisodeLog）包含以下信息：

- **pk**: 对话的唯一ID
- **environment**: 环境ID
- **agents**: 参与对话的代理列表
- **tag**: 对话标签（用于分组和查询）
- **models**: 使用的模型列表 [env_model, agent1_model, agent2_model, ...]
- **messages**: 对话消息，按轮次组织
  - 每一轮包含多个消息
  - 每条消息格式: (发送者, 接收者, 消息内容)
- **reasoning**: 对话的推理/总结
- **rewards**: 每个代理的奖励分数列表

## 对话存储位置

### 如果使用Redis后端：
对话保存在Redis数据库中。需要Redis实例正常运行。

### 如果使用本地存储：
对话保存在本地文件系统中。具体位置由sotopia的存储配置决定。

## 修改日志说明

### `server.py` 的修改

在 `arun_one_episode` 函数中，修改了epilog创建逻辑：

1. **处理缺失的profile**：
   - 如果 agent 或 env 没有 profile，使用 agent_name 或 "default_env" 作为备选
   
2. **处理缺失的评分字段**：
   - 如果 info 中没有 "complete_rating"，默认使用 0.0
   
3. **更好的错误处理**：
   - 用 try-except 包装epilog创建，避免保存失败时中断程序
   - 失败时输出警告日志而不是抛出异常

## 常见问题

### Q: 为什么epilog创建失败？
A: 检查：
1. 环境变量 `CUSTOM_API_KEY` 是否正确设置
2. 模型服务是否正常运行
3. 数据库（Redis或本地存储）是否可用

### Q: 如何更改保存的标签？
A: 修改 `minimalist_demo.py` 或脚本中的 `tag` 参数：
```python
run_async_server(
    ...,
    tag="my_custom_tag",  # 改为你想要的标签
    push_to_db=True,
)
```

### Q: 如何清除已保存的对话？
A: 如果使用本地存储，可以直接删除数据文件。如果使用Redis，使用Redis CLI删除相应的key。

### Q: 为什么某些信息显示为默认值？
A: 当聊天过程中某些可选字段（如complete_rating、profile.pk）不可用时，使用了默认值。这样可以确保对话记录仍然被保存，不会因为缺少可选字段而失败。

## 下一步

现在你可以：
1. 运行 `python examples/minimalist_demo.py` 来生成对话记录
2. 使用 `python examples/view_episodes.py` 来查看保存的对话
3. 根据需要修改标签和其他参数
