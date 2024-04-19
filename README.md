# embedding-server

Title: Creating an OpenAI Embedding API Compatible Server

Purpose:
- For embedding models to generate well-separated embedding vectors, they need to be trained with a good tokenizer and a large amount of data.
- However, finding a suitable embedding model for multi-language environments can be challenging.
- The quality of the embedding model that can be obtained depends on the appropriate level of the model's performance.

Limitations:
- When extracting and creating an embedding model, there are inherent limitations.

Process:
## Model Extraction
- Use mergekit to extract the 0th layer.
- Typically, this extraction includes the embed_tokens layer along with the RMSNorm layer.
- Use the following command to extract the model using mergekit:
```
# mergekit [config_file] [save_target_path]
mergekit-yaml ./example.yaml ./test
```

## Embedding Extraction
- The `embedding` function in `server.py` handles the embedding extraction. It loads the model using the `AutoModel` and `AutoTokenizer` from the `transformers` library and processes the tokens.
- The embeddings are extracted by processing the tokens through the loaded layers.
- However, positional encoding is not separately handled in this process.
- Since other embedding models do not require appropriate positional encoding, the code for it is not included.

## Model Configuration
- In the `server.py` code, you need to set the model name and path:
```
models_list = {
  # Model Name: Model Path
  'text-embedding-3-small': './test/'
}
```
- The following code in the script sets the default model name:
```
model = data.get('model', 'text-embedding-3-small')
```

## Execution & Testing
```
# Start the server
python server.py

# Test
curl -X POST -H "Content-Type: application/json" -d '{"input": "hello", "model": "text-embedding-3-small"}' http://localhost:5000/v1/embeddings
```

Required Libraries:
- Install the necessary libraries using the following command:
```
pip install flask torch transformers
```

License:
- MIT

Warning: Testing
- The code has been tested for functionality, but the quality of the embeddings has not been evaluated.
- The following models have been tested:
  - LLaMa v2 / v3
  - Gemma 7B 1.1
  - Mistral 7B

References:
- https://www.sbert.net/docs/pretrained_models.html
- https://platform.openai.com/docs/guides/embeddings
