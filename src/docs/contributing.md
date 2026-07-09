# 贡献指南

欢迎为 TokenCode 贡献代码！以下是参与方式。

## 提交 Issue / Pull Request

- Issue 与 Pull Request 请提交至 [miuiadmin/tokencode](https://github.com/miuiadmin/tokencode)。
- 提交前请先搜索是否已有相同 Issue，避免重复。

## 开发流程

1. Fork 仓库并创建分支。
2. 从源码构建的方式见 [安装与构建](./install.md)。
3. 改动后运行格式化与测试：

   ```shell
   just fmt
   just fix -p <你修改的 crate>
   just test -p <你修改的 crate>
   ```

4. 提交 Pull Request，并在描述中写清 **动机（为什么改）** 与 **净变更（改了什么）**。

## DCO（Developer Certificate of Origin）

提交时请在 commit message 末尾追加一行：

```
Signed-off-by: 你的名字 <邮箱>
```

这等价于 DCO，表明你拥有该贡献的提交权；无需额外签署协议。

## 行为准则

请保持友善、尊重的交流态度。
