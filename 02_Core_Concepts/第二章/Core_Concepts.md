现代大语言模型的训练通常分为**预训练（Pre-training）** 和**后训练（Post-Training）**
![[螢幕截圖 2026-05-08 21.45.03.png]]

1.预训练阶段将海量文本序列作为输入，模型通过"预测下一个词"（Next-Token Prediction）学习世界知识和语言规律。这是代价最高的一步，赋予了模型基础能力。

- **输入**：维基百科、书籍、网页代码等无标签纯文本。
- **目标**：最小化语言模型在下一个词预测任务上的交叉熵损失。
- **结果**：具备语言能力和世界知识，但无法以对话方式回应的基座模型（Base Model）。
- **数据示例（纯文本）**：
    
    ```
    {
      "text": "巴黎是法国的首都，位于法国北部巴黎盆地的中央，建都已有1400多年的历史..."
    }
    ```
    

就是只会基础常识，能预测下一个词，但是不会对话。

2.### 督微调（Supervised Fine-Tuning, SFT）

监督微调阶段将高质量的问答对作为输入，让模型学会以对话格式回应人类。

- **输入**：人工编写的 (Prompt, Response) 对。
    
- **目标**：在问答数据上继续做 Next-Token Prediction，使模型模仿人类的回答格式。
    
- **结果**：一个能以对话方式交互的指令模型（Instruct Model）。
    
- **数据示例（问答对）**：
    
    ```
    {
      "messages": [
        { "role": "user", "content": "请问法国的首都是哪里？" },
        {
          "role": "assistant",
          "content": "法国的首都是巴黎。它位于法国北部巴黎盆地的中央。"
        }
      ]
    }
    ```

学会了对话，会回答，只是模仿，不知道什么是好的答案，什么是坏的答案。

一个经过 SFT 的模型可能学会了对用户说"你好"，却在用户说"学数学没用"时盲目附和，而不是礼貌地纠正；

3.### 强化学习与对齐（RL / Alignment）

- **输入**：(Prompt, Chosen, Rejected) 三元组。
- **目标**：最大化好回答与坏回答的概率差。
- **结果**：回答质量更高、更安全的对齐模型（Chat Model）。
- **数据示例（偏好三元组）**——典型的偏好对比数据格式（与 [1-generate_data.py](https://walkinglabs.github.io/code/chapter02_dpo/1-generate_data.py) 生成的格式一致）：
    
    ```
    {
      "prompt": "我今天心情很差，不想上班。",
      "chosen": "听到你这么说我很难过。如果觉得压力太大，适当请个假休息一天也是可以的，身体和心理健康永远排在第一位。",
      "rejected": "不上班你吃什么？赶紧去工作！别矫情了。"
    }
    ```
    

回到我们在上一节中运行的 [3-train_dpo.py](https://walkinglabs.github.io/code/chapter02_dpo/3-train_dpo.py)，代码里加载的 `preference_data.json` 正是这种格式。数据加载部分将 JSON 文件解析为 `prompt`、`chosen`、`rejected` 三个字段，分别对应符号表中的 xx、ywyw​、ylyl​：

```
data_dict = {
    "prompt": [item["prompt"] for item in data_list],    # → x
    "chosen": [item["chosen"] for item in data_list],     # → y_w
    "rejected": [item["rejected"] for item in data_list]  # → y_l
}
train_dataset = Dataset.from_dict(data_dict)
```

值得注意的是，三个阶段的数据形态发生了根本性的变化：第一阶段是纯文本，没有标注；第二阶段是问答对，告诉模型"好的回答长什么样"；第三阶段是好坏对比，告诉模型"好的回答比坏的好在哪里"。从"预测下一个词"到"模仿人类回答"再到"区分好坏"——每一步都在让模型的行为更贴近人类的期望。



