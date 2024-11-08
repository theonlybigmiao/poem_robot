# ChatBotX

本项目是基于Hugging Face的tansformers库训练出的对诗模型，受启发于另一个使用seq2seq模型构建的单轮对话模型(https://github.com/Schellings/Seq2SeqModel)。目前本地训练时长已超一百小时，但效果一般（具体效果如下图所示），待进一步完善...

<img src="E:/poem_robot/效果.png" width="60%">

## 项目结构

```shell
│  .gitignore
│  README.md
│  requirements.txt
│  train.py
│  main.py
|  requirements.txt
|
├─data
│   poem.txt
│
├─finetuned_model
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

训练好的模型会保存在`finetuned_model`目录下。

- 体验交互对诗

```
python main.py
```


