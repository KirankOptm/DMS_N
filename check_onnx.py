import sys, onnxruntime as ort

path = sys.argv[1]
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("Model:", path)
for i in sess.get_inputs():
    print("Input:", i.name, i.shape, i.type)
for o in sess.get_outputs():
    print("Output:", o.name, o.shape, o.type)