````

这份 `AGENT.md` 的重点是把 Agent 限制成“工程执行者”，不是“方法发明者”。你之后给 Codex 的每个任务最好也写成这种格式：

```text
只修改 xxx.py。
目标：实现 xxx。
不要修改训练脚本。
不要新增 loss。
不要新增依赖。
完成后做 dummy forward 测试。
````