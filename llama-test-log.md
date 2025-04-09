记录一下配置llama遇到的问题（目前没时间继续下去了），只能一段时间后resume了。
1. 修改了一些文件中为绝对导入。
2. 添加了boolq的数据模板
3. 添加了llama模型，mistral的模板完全可以用到llama上，自己重新写很麻烦。一开始llama7b是没有chat模板的，用了llama2 7b-chat的模板，但是效果非常差。
4. 添加了llama在shift_encoder和shift_model中以及train中的部分。
5. 由于llama是语言模型，可能ICLTestbed中修改了代码，必须在处理input的时候需要image和text。这里暂时用image=[]通过。ICLTestbed中的custom_model_tutorial中用mistral模型的自适应的教程也不再适用。
5. 遇到的最大问题就是llama是没有tokenizer的，也没有pad_token。需要直接用processor以及eos_token当作pad_token。
6. shift_model.py的process_input函数对 dtype=self.dtype 不再接收，需要在模型的process_input中修改处理逻辑，或者删除了（我的做法）。
7. 照着tutorial中custom_model中处理，以及经历了上述的步骤后，代码可以运行，但是在训练的时候部分epoch直接loss变成inf了， 并且跑到一半显示 CUDA out of memory了（4*3090）

一些结果：
在jupyter进行小的测试，成功加载llama7b在boolq上的正确率是0.6左右（未mimic训练及任何微调的）。DoRA中论文的是llama7b微调后效果大概在0.7。
