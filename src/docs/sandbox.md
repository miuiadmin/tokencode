## 沙箱与审批

TokenCode 通过操作系统级沙箱约束智能体的文件与网络操作，并以审批模式决定何时需要用户确认。

- **沙箱后端**：macOS 使用 seatbelt，Linux 使用 landlock / bubblewrap。
- **审批模式**：可在配置中指定默认审批策略（如只读、工作区可写、需逐次确认等）。

细粒度的命令与路径规则见 [执行策略](./execpolicy.md)。
