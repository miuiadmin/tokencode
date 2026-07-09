# 安全策略

感谢你帮助 TokenCode 保持安全！

## 报告安全漏洞

如果你发现安全漏洞，请负责任地通过以下渠道私下报告：

- **首选**：GitHub Security Advisories —— [提交 advisory](https://github.com/miuiadmin/tokencode/security/advisories/new)
- **备选**：邮件至 [miui@outlook.sg](mailto:miui@outlook.sg)

请不要在公开 Issue 中披露未修复的安全漏洞。

## 安全边界

TokenCode 通过操作系统级沙箱（macOS seatbelt、Linux landlock / bubblewrap）、审批模式与网络控制来约束智能体的文件与网络行为。详见 [沙箱与审批](./docs/sandbox.md)。
