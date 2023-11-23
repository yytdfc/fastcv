import io
import numpy as np
import cv2
import torch
import onnx
# import onnxruntime

class OnnxModel:
    def __init__(self, model_path):
        self.onnx_model = onnx.load(model_path)
        self.reload_session(640, 640)

    def reload_session(self, h, w):
        print("Deblur Reload onnx:", h, w)
        self.onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = h
        self.onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = w
        self.onnx_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = h
        self.onnx_model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = w
        with io.BytesIO() as f:
            onnx.save(self.onnx_model, f)
            f.seek(0)
            self.ort = onnxruntime.InferenceSession(f.getvalue())
        self.h = h
        self.w = w

    def __call__(self, x):
        if x.ndim == 3:
            t = np.expand_dims(x.transpose(2, 0, 1), 0).astype(np.float32) / 255.
        else:
            t = x
        # print(t.shape)
        if self.h!=t.shape[2] or self.w!=t.shape[3]:
            self.reload_session(t.shape[2], t.shape[3])
        # print(self.onnx_model.graph.input[0], dir(self.onnx_model.graph.input[0]))
        ort_inputs = {self.ort.get_inputs()[0].name: t}
        out = self.ort.run([], ort_inputs)[0]
        if x.ndim == 3:
            out = np.clip(out[0].transpose(1, 2, 0) * 255., 0, 255).round().astype(np.uint8)
        return out
