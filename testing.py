# pip install onnxruntime --break-system-packages

import onnxruntime as ort

session = ort.InferenceSession("best_v2.onnx", providers=["CPUExecutionProvider"])
print("ONNX model loaded successfully!")
