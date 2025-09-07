# mutinode-distributed-gpu-inference
Meant to be used short term as essentially a local gpu cluster on the NERVE robotics piplines. These require various models/algorithms to be trained on GPUs and often run into Cuda Out of Memory errors.
<br />
<br />
Controllable through config.json.
<br />
<br />
I will start with attempting to get SAM2 to work across multiple GPUs. The 
<br />
<br />
Future: customize tensor splits with cuda code in extensions.