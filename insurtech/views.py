from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import InsuraTransformer, Lang
from .key_map import key_mapping
import torch #torch version 2.1.0+cpu
import torch.nn as nn
import torch.optim as optim
import json
import os


# Create your views here.
def chat(request):
    return render(request, 'chat.html')





"""Chatbox functions goes here"""


english_lang = Lang("english_language")
UNK_token = 2


def tokenize_sentence(sentence, lang_instance):
    token_ids = []
    for word in sentence.split():
        if word in lang_instance.word2index:
            token_ids.append(lang_instance.word2index[word])
        else:
            token_ids.append(UNK_token)  # Handle unknown words
    return token_ids


def process_user_message(user_message):
    print('User Message: ', user_message)
    tokenized_src = tokenize_sentence(user_message, english_lang)
    # Assuming tgt is also needed, tokenizing an empty string as an example
    tokenized_tgt = tokenize_sentence("", english_lang)  # Change "" to the target sentence

    src = torch.tensor(tokenized_src).long().unsqueeze(0)  # Convert to tensor and add batch dimension
    tgt = torch.tensor(tokenized_tgt).long().unsqueeze(0)  # Convert to tensor and add batch dimension
    
    print('src: ', src)
    print('tgt: ', tgt)
    print('*'*100)

    return src, tgt


def map_keys(mapping, key):
    return mapping.get(key, key)

# @csrf_exempt
# def ai_response(request):
#     print('='*100)
#     print('torch version:')
#     print(torch.__version__)
#     print('='*100)

#     user_message = request.POST.get('userMessage', '')
#     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insura_transformer_model.pth')

#     if not os.path.exists(model_path):
#         response = {'reply': 'No Model found in the directory'}
#     else:
#         try:
#             # Create an instance of your model class after loading the checkpoint
#             model = InsuraTransformer(src_vocab_size=955, tgt_vocab_size=5155, d_model=512,
#                                        num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)

#             model_optimizer = optim.Adam(model.parameters(), lr=0.001)

#             checkpoint = torch.load(model_path, map_location='cpu')

#             # Create a new state_dict by mapping keys using key_mapping
#             new_state_dict = {}
#             for key, value in model.state_dict().items():
#                 mapped_key = key_mapping.get(key, key)  # Use the original key if not found in key_mapping
#                 new_state_dict[key] = checkpoint['state_dict'].get(mapped_key, value)

#             loaded_keys = set(new_state_dict.keys())
#             expected_keys = set(model.state_dict().keys())

#             missing_keys = expected_keys - loaded_keys
#             unexpected_keys = loaded_keys - expected_keys

#             model.load_state_dict(new_state_dict)

#             # Create a new optimizer instance with the same parameters as the original optimizer
#             model_optimizer = optim.Adam(model.parameters(), lr=0.001)

#             # Load the optimizer state from the checkpoint
#             model_optimizer.load_state_dict(checkpoint['optimizer'])

#             model.eval()

#             src, tgt = process_user_message(user_message)

#             with torch.no_grad():
#                 # Generate masks (modify this based on your model's input requirements)
#                 src_mask, tgt_mask = model.generate_mask(src, tgt)

#                 # Adjust this line based on your model's input requirements
#                 response = model(src, tgt)
#                 print('Response:', response)
#                 print('*'*100)
#                 print("Response Shape:", response.shape)
#                 print('*'*100)

#                 # Convert the PyTorch tensor response to a JSON serializable format
#                 response_numpy = response.cpu().detach().numpy()  # Convert tensor to NumPy array
#                 response_json = json.dumps(response_numpy.tolist())  # Convert NumPy array to JSON
#                 print("Response Numpy:", response_numpy)
#                 print('*'*100)
#                 print("Response JSON:" ,response_json)
#                 # Format the AI response as needed
#                 response = {'reply': response_json}
#                 print('i am here')

#             return JsonResponse(response)

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             error_response = {'reply': f'Error: {str(e)}'}
#             print('i am here with error')
#             return JsonResponse(error_response, status=500)


@csrf_exempt
def ai_response(request):
    user_message = request.POST.get('userMessage', '')
    print('ai_response User Message:' ,user_message)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insura_transformer_model.pth')

    if not os.path.exists(model_path):
        response = {'reply': 'No Model found in the directory'}
    else:
        try:
            model = InsuraTransformer(src_vocab_size=955, tgt_vocab_size=5155, d_model=512,
                                       num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)

            model_optimizer = optim.Adam(model.parameters(), lr=0.001)

            checkpoint = torch.load(model_path, map_location='cpu')

            new_state_dict = {}
            for key, value in model.state_dict().items():
                mapped_key = key_mapping.get(key, key)
                new_state_dict[key] = checkpoint['state_dict'].get(mapped_key, value)

            loaded_keys = set(new_state_dict.keys())
            expected_keys = set(model.state_dict().keys())

            model.load_state_dict(new_state_dict)

            model_optimizer = optim.Adam(model.parameters(), lr=0.001)
            model_optimizer.load_state_dict(checkpoint['optimizer'])

            model.eval()

            src, tgt = process_user_message(user_message)

            with torch.no_grad():
                src_mask, tgt_mask = model.generate_mask(src, tgt)
                response = model(src, tgt)

                response_numpy = response.cpu().detach().numpy()
                response_json = json.dumps(response_numpy.tolist())
                response = {'reply': response_json}

            return JsonResponse(response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {'reply': f'Error: {str(e)}'}
            return JsonResponse(error_response, status=500)
