# ChatBotX

本项目是基于transformers构建的对诗模型。受启发于另外一个使用seq2seq模型构建的[单轮对话模型](https://github.com/Schellings/Seq2SeqModel)。

目前模型的效果一般，待进一步完善...

<img src="E:/poem_robot/效果.png" alt="模型效果展示" style="width:60%;" />

## 项目结构

```shell
│  .gitignore
│  README.md
│  requirements.txt
│  train.py
│  main.py
│
├─data
│      poem.txt
│
├─finetuned_models
```

## 环境配置

环境配置需要两个步骤：

- 下载项目文件
- 安装相关的第三方库

具体步骤如下：

```
git@github.com:theonlybigmiao/poem_robot.git && cd poem_robot
pip install requirements.txt
```

## 模型使用

- 模型训练

```
python train.py
```


- 模型预测

训练好的模型会保存在`finetuned_models`目录下。

- 体验交互式对诗

```
python main.py
```

