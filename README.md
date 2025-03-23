# medical-inference
Inference Server using HuggingFace to output medical diagnoses based on medical notes.

# Model Choice
We can use a decoder style model and specify the input and output format. BioGPT works with zero shot learning and is medically aware to provide diagnoses. Encoder style models will require predefined labels making them less flexible. Decoder model will not require a fixed label set and work better with rare diseases.

After picked a suitable model for clinical text diagnosis. I then set up an inference server using Huggingface rather than vLLM due to simplicity of keeping everything in the huggingface endpoint. We are still able to leverage scalability using this inference setup.

I used FastAPI to create the HTTP endpoint that our model will be exposed to for easy input prompting. 

# Input / Output Parsing
The input to the model is a json where the clinical notes is the input string. I focused more on formatting the output by prompting the model to output a comma separated list. I then processed the output using Pydantic to ensure strong type checking. 

Also engineered Dockerfile to install the model such that we do not have to download it everytime we run file.

# Possible Improvements
- Async Inference - convert endpoint to be async for non-blocking operations for better performance under heavy load (use await keyword)
- Use of data parallel / model parallel (multi-node inference)
- Caching for repeated clinical notes to speed up inference
- Include batch processing for improved throughput
- Quantize model for faster and cheaper inference
- Have dynamic lengths to output based on the size of input
