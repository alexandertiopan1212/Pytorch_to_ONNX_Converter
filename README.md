# 🧠 Pytorch to ONNX Converter

Convert your trained PyTorch model into the portable and interoperable ONNX format — and even prepare it for TensorFlow integration!

---

## 🚀 Overview

This project demonstrates how to:

- ✅ Build a simple Convolutional Neural Network (CNN) in PyTorch
- 🔁 Export the trained `.pth` model to ONNX format
- 🛠 Modify ONNX input names if needed
- 🔄 (Optional) Prepare ONNX for TensorFlow conversion using `onnx-tf`

The ONNX format enables **framework interoperability** — use your PyTorch models with TensorFlow, TensorRT, and other inference engines.

---

## 🧰 Dependencies

Make sure the following packages are installed:

```bash
pip install torch onnx onnx-tf tensorflow tensorflow-probability
```

---

## 🧠 CNN Model Architecture

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 32 * 7 * 7)
        return self.block3(x)
```

---

## 🧪 Model Conversion Steps

### 1. Load and Prepare PyTorch Model

```python
pytorch_model = CNN()
pytorch_model.load_state_dict(torch.load("model.pth"))
pytorch_model.eval()
```

### 2. Export to ONNX

```python
dummy_input = torch.randn(5, 3, 28, 28)
torch.onnx.export(pytorch_model, dummy_input, "model.onnx")
```

### 3. Rename ONNX Input (if needed)

```python
import onnx
from onnx import helper

onnx_model = onnx.load("model.onnx")
name_map = {"input.1": "input_1"}
new_inputs = []

for inp in onnx_model.graph.input:
    if inp.name in name_map:
        new_inp = helper.make_tensor_value_info(
            name_map[inp.name],
            inp.type.tensor_type.elem_type,
            [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        )
        new_inputs.append(new_inp)
    else:
        new_inputs.append(inp)

onnx_model.graph.ClearField("input")
onnx_model.graph.input.extend(new_inputs)

for node in onnx_model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in name_map:
            node.input[i] = name_map[input_name]

onnx.save(onnx_model, "model-new.onnx")
```

---

## 📦 Output

- `model.onnx` : Exported ONNX model
- `model-new.onnx` : ONNX model with renamed input node (if applicable)

---

## 🧩 Future Work

- TensorFlow `.pb` conversion support
- ONNX inference sample with real image input
- Web deployment demo using ONNX.js

---

✅ Simple, robust, and ready for your deployment pipeline.
