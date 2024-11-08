from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import re
import time
# 加载微调后的模型和分词器
# model_path="./results/checkpoint-2000"
model_path = "./finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 第一种处理：重复原来下标为3的字
def process_text(res):
    # 以"。"进行分隔
    parts = res.split("。")
    # 第四个"。"后的所有内容被舍弃
    processed_parts = parts[:4]

    # 以"，"进行分隔
    for i in range(len(processed_parts)):
        words = processed_parts[i].split("，")
        for j, word in enumerate(words):
            if len(word) == 6:  # 如果该元素长度为6个汉字
                # 在中间位置加上一个"，"
                words[j] = word[:3] + word[3] + word[3:]
        # 重新组合句子
        processed_parts[i] = "，".join(words)

    # 最后返回处理好的res
    return "。".join(processed_parts)

# 第二种处理：中间加逗号隔开
def process_text_2(res):
    # 以"。"进行分隔
    parts = res.split("。")
    # 第四个"。"后的所有内容被舍弃
    processed_parts = parts[:4]

    # 以"，"进行分隔
    for i in range(len(processed_parts)):
        words = processed_parts[i].split("，")
        for j, word in enumerate(words):
            if len(word) == 6:  # 如果该元素长度为6个汉字
                # 在中间位置加上一个"，"
                words[j] = word[:3] + '，' + word[3:]
            elif len(word)==4:
                words[j] = word[:2]+"忽"+word[2:]
        # 重新组合句子
        processed_parts[i] = "，".join(words)

    # 最后返回处理好的res
    return "。".join(processed_parts)


def extract_between_cls_sep(text):
    match = re.search(r"\[CLS\](.*?)\[SEP\]", text)

    if match:
        return match.group(1).strip()
    else:
        return None

# # 输入文本
# # input_text = "折枝空度花难渡，"


# # inputs = tokenizer(
# #     input_text,
# #     return_tensors="pt",
# #     return_attention_mask=True,
# #     return_token_type_ids=False,
# #     add_special_tokens=True,
# # )

# # # 使用模型生成文本

# # # output = model.generate(
# # #     inputs["input_ids"],
# # #     attention_mask=inputs["attention_mask"],
# # #     max_length=100,
# # #     do_sample=True,
# # #     num_return_sequences=3,
# # #     top_k=3,
# # #     top_p=0.8,
# # #     temperature=0.7,
# # #     repetition_penalty=2.0,
# # # )

# # output = model.generate(
# #     inputs["input_ids"],
# #     attention_mask=inputs["attention_mask"],
# #     max_length=70,
# #     do_sample=False,
# # )

# # # 输出结果
# # for i, beam in enumerate(output):
# #     print(f"Beam {i+1}:")
# #     res=tokenizer.decode(
# #             beam, skip_special_tokens=True, clean_up_tokenization_spaces=True
# #         ).replace(" ", "")
# #     res=process_text(res).strip()+'。'

title = gr.HTML("<center>简易诗词创作模型</center>")
with gr.Blocks() as demo:
    with gr.Row():
        title.render()
    msg = gr.Textbox(placeholder="请输入一句诗")
    chatbot = gr.Chatbot()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        message="[CLS]"+message+'，'
        inputs=tokenizer(
            message,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=70,
            do_sample=False,
        )
        for i, beam in enumerate(output):
            print(f"Beam {i+1}:")
            res=extract_between_cls_sep(tokenizer.decode(
                beam, skip_special_tokens=False
            ).replace(" ", "")).strip()
            chat_history.append((message, res))
            time.sleep(1)
            return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)



if __name__ == '__main__':
    demo.launch(share=True)


# import random
# import time
# import gradio as gr
#
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button([msg, chatbot])
#
#     def respond(message, chat_history):
#         bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#         chat_history.append((message, bot_message))
#         time.sleep(2)
#         return "", chat_history
#
#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
#
# demo.launch()