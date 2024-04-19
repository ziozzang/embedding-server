from flask import Flask, request, jsonify
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

models_list = {
  # Model Name: Model Path
  'text-embedding-3-small': './test/'
}
models_loaded = {
}

def _load_models():
  for i in models_list:
    if not ( i in models_loaded):
      models_loaded[i] = [
          AutoTokenizer.from_pretrained(models_list[i]),
          AutoModel.from_pretrained(models_list[i])
        ]

# Loading Models
_load_models()

def embedding(model_name, text):
  # Get Tokenizer and Models
  tokenizer, model = models_loaded[model_name]

  # Tokenizing
  inputs = tokenizer(text, return_tensors='pt')
  token_len = len(inputs['input_ids'][0])

  # Get Embedding
  with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state

  # Token len == embedding count -> after encoder, embedding layers + positional encoding => attention
  #token_len = len(embeddings[0])

  #embeddings = F.normalize(embeddings, p=2, dim=1)
  
  # Embedding squezzing into single array.
  text_embedding = torch.mean(embeddings, dim=1).squeeze()
  # Ref: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
  # mean_pooling has no positional encoding. :(

  return (token_len, text_embedding.tolist())


@app.route('/v1/embeddings', methods=['POST'])
def get_embeddings():
    data = request.get_json()
    text = data.get('input', '')
    model = data.get('model', 'text-embedding-3-small')

    if not text:
        return jsonify({'error': 'No input text provided'}), 400

    start_time = time.time()
    token_len, embeddings = embedding(model, text)
    end_time = time.time()
    execution_time = end_time - start_time
    #print(f"execution time: {execution_time:.5f} 초")

    response = {
        'object': 'list',
        'data': [
            {
                'object': 'embedding',
                'index': 0,
                'embedding': embeddings
            }
        ],
        'model': model,
        'usage': {
            'prompt_tokens': token_len,
            'total_tokens': token_len
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

