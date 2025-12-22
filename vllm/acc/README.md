参考示例：
```shell
evalscope eval   \
    --model DeepSeek-R1-Distill-Llama-70B   \
    --api-url http://127.0.0.1:8000/v1   \
    --api-key EMPTY   \
    --eval-type openai_api   \
    --datasets math_500   \
    --eval-batch-size 32
```

参考：
- WangKang's Blog：https://dayangya.notion.site/EvalScope-2920f422eb85809e87f3f67cf5a54a4d?source=copy_link
- EvalScope: https://evalscope.readthedocs.io/zh-cn/latest/index.html