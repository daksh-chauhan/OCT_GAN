# Image Generation with ONNX and PyTorch

## Overview
This project involves training a deep learning model for image generation using PyTorch and exporting it to ONNX format for inference. The focus is on both training and utilizing custom-trained models.

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install onnxruntime torch torchvision matplotlib numpy tqdm
```

## Usage

### Training the Model
The model is trained using PyTorch. Ensure your dataset is correctly structured and modify training parameters as needed.

### Inference with ONNX
After training, the model is exported to ONNX and can be used for image generation:

```python
# Example usage
inspect_onnx_model("Best models/model.onnx")
generate_images("Best models/model.onnx")
```

### Model Inspection
The ONNX model can be examined to check its input and output properties:

```python
inspect_onnx_model("path/to/model.onnx")
```

## Notes
- The generated images have a resolution of **128x128**.
- Ensure you have a GPU for faster execution (`CUDA` support is detected automatically).
- Modify batch size and training parameters as needed.
- The dataset should be structured correctly for `torchvision.datasets.ImageFolder`.

