# FashionMNIST-onnx

## 转换器使用方法

需安装tensorflow 2.x和tf2onnx包。

替换[原仓库](https://github.com/LC-John/Fashion-MNIST)code目录中的同名文件，再运行`test.py`即可。

## 模型使用方法

需安装onnxruntime包。

```python
def load_model():
    sess = onnxruntime.InferenceSession(ONNX_PATH)

    def inference(x: np.ndarray):
        # We use {dim: BxCxWxH, range [0,1]}
        # Model requires {dim: BxWxHxC, range [0,255]}
        tx = np.transpose(x, axes=(0, 2, 3, 1)) * 255.0
        return sess.run([], {'fmnist_cnn/Placeholder:0': tx})[0]

    def inference_torch(x: torch.Tensor):
        tx = x.detach().cpu().numpy()
        result = inference(tx)
        return torch.from_numpy(result)

    return inference, inference_torch
```
