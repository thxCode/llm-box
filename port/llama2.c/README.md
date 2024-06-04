# llama2.c

> tl;dr Go port of the [karpathy/llama2.c](https://github.com/karpathy/llama2.c).

[karpathy/llama2.c#run.c](https://github.com/karpathy/llama2.c/blob/master/run.c) is a 700-line C file that can
inference Llama 2-like LLMs,
[run.go](./run.go) is a Go imitation.

## Build

```shell
make -C ../.. llama2.c
```

## Run

First, download the model checkpoints, see https://github.com/karpathy/llama2.c#models.

By default, the [tok512.bin](./tok512.bin) tokenizer is
for [stories260K.bin](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) checkpoint, you can easily kick
off with the following script.

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K.bin

./run stories260K.bin -z tok512.bin -i "Once upon a time, there was a"

```

When you want to try larger parameters checkpoints, you have to switch the tokenizer to [tokenizer.bin](./tokenizer.bin).

For example, you can try `stories42M.bin` model as below.

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin

./run stories42M.bin -z tokenizer.bin -i "One day, Tesla met a tiger"

```

## LICENSE

MIT
