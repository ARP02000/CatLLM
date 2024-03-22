# whisper-large-finetune

## 运行步骤
1. 利用anaconda 执行命令 `pip install -r requirements.txt` 安装对应的环境 (能装好环境就行)
2. 执行 `python aishell.py` 下载aishell数据集，如数据data_aishell.tgz/data_aishell.gz已经在本地，则在文件中配置对应的文件路径，同时确定解压目录
2. 执行 `python finetune.py` 利用aishell数据集对whisper-large-v2进行训练，模型可以在huggingface找到openai官方下载预训练的权重文件(huggingface目前在国内被墙的厉害)
3. 执行 `python merge_lora.py` 将权重文件合并
4. 测试环境下，推理可以通过执行命令 `python infer_tfs.py` 进行，注意修改好要推理的音频文件路径
5. `infer_ct2.py` 依赖于转换后的CTranslate2模型，具体见注意事项部分
6. `infer_server.py` 同样依赖于转换后的CTranslate2模型，如不需要部署到线上可忽略

## 注意事项
1. 如不想进行fine-tune,仅需要在 `infer_tfs.py` 中将模型设置为openai提供的官方模型，同样能够实现推理过程
2. 如需加速推理，则需先执行命令 `ct2-transformers-converter --model (your_model_path) --output_dir (your_output_path) --copy_files tokenizer.json preprocessor_config.json --quantization float16`
3. 理论上2中的命令对openai提供的官方模型同样有效，但本人并未尝试
4. 显存小于20GB的建议fine-tune更小的模型

