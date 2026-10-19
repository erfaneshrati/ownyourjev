import time, sys, numpy as np, torch, onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
D="/tmp/claude-1000/-media-erfan-Data-ownyourjev/887ffe38-400c-4edb-bb16-1a88de00b3d4/scratchpad"
hf="Alibaba-NLP/gte-modernbert-base"
tok=AutoTokenizer.from_pretrained(hf)
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m; s.h=torch.nn.Linear(768,20)
    def forward(s,i,a): return torch.softmax(s.h(s.m(input_ids=i,attention_mask=a).last_hidden_state[:,0]).view(-1,4,5),-1)
for nl in [int(x) for x in sys.argv[1].split(",")]:
    m=AutoModel.from_pretrained(hf,dtype=torch.float32,attn_implementation=os.environ.get("ATTN","sdpa"))
    m.layers=m.layers[:nl]; m.config.num_hidden_layers=nl
    w=W(m).eval()
    enc=tok(["hello world"]*2,padding=True,return_tensors="pt")
    p=f"{D}/lat{nl}.onnx"
    torch.onnx.export(w,(enc["input_ids"],enc["attention_mask"]),p,input_names=["input_ids","attention_mask"],output_names=["probs"],dynamic_axes={"input_ids":{0:"b",1:"s"},"attention_mask":{0:"b",1:"s"}},opset_version=17,dynamo=False)
    q=p.replace(".onnx",".int8.onnx"); quantize_dynamic(p,q,weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=["MatMul","Gemm"])
    o=ort.SessionOptions(); o.intra_op_num_threads=4
    s=ort.InferenceSession(q,o,providers=["CPUExecutionProvider"])
    for L in [int(x) for x in os.environ.get("LENS","256,384,512,768").split(",")]:
        ids=np.random.randint(1000,5000,(1,L)).astype(np.int64); am=np.ones_like(ids)
        t=[]
        for k in range(40):
            t0=time.perf_counter(); s.run(["probs"],{"input_ids":ids,"attention_mask":am}); t.append((time.perf_counter()-t0)*1000)
        t=t[3:]; print(f"layers={nl} L={L} p50={np.percentile(t,50):.1f} p95={np.percentile(t,95):.1f}",flush=True)
