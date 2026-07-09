# 非交互执行（tokencode exec）

`tokencode exec` 以非交互方式运行单轮任务：给定一个提示，智能体执行并直接输出结果，适合在脚本或流水线中调用。

```shell
tokencode exec "解释这段代码库"
```

非交互模式默认 `RUST_LOG=error`，消息内联打印，无需监控额外日志文件。
