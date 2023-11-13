from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import InsuraTransformer
import torch #torch version 2.1.0+cpu
import torch.nn as nn
import torch.optim as optim
import json
import os


# Create your views here.
def chat(request):
    return render(request, 'chat.html')


def process_user_message(user_message):
    src = torch.tensor([1, 2, 3])
    tgt = torch.tensor([4, 5, 6])

    return src, tgt


# def ai_response(request):
#     user_message = request.POST.get('userMessage', '')

#     # Define the model file path
#     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insura_transformer_model.pth')

#     # Check if the model file exists
#     if not os.path.exists(model_path):
#         response = {'reply': 'No Model found in the directory'}
#     else:
#         try:
#             # Load the model checkpoint
#             checkpoint = torch.load(model_path, map_location='cpu')

#             # Create an instance of your model class
#             model = InsuraTransformer(src_vocab_size=10000, tgt_vocab_size=5000, d_model=512,
#                                        num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)

#             # Load the model state_dict
#             model.load_state_dict(checkpoint['state_dict'])

#             # If you have an optimizer and want to load its state_dict as well
#             if 'optimizer' in checkpoint:
#                 # Define and initialize your optimizer before loading its state_dict
#                 optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#                 optimizer.load_state_dict(checkpoint['optimizer'])

#             model.eval()

#             # Process the user's message and generate an AI response
#             with torch.no_grad():
#                 # Generate masks (modify this based on your model's input requirements)
#                 src_mask, tgt_mask = model.generate_mask(src, tgt)

#                 # Adjust this line based on your model's input requirements
#                 response = model(src, tgt)

#             response = {'reply': response}  # Modify this to format the AI response as needed
#         except Exception as e:
#             response = {'reply': f'Error: {str(e)}'}

#     return JsonResponse(response)



# @csrf_exempt  # This is added to disable CSRF protection, use it carefully
# def ai_response(request):
#     user_message = request.POST.get('userMessage', '')
#     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insura_transformer_model.pth')
    
#     if not os.path.exists(model_path):
#         response = {'reply': 'No Model found in the directory'}
#     else:
#         try:
#             # Create an instance of your model class after loading the checkpoint
#             model = InsuraTransformer(src_vocab_size=10000, tgt_vocab_size=5000, d_model=512,
#                                         num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)
            
#             model_optimizer = optim.Adam(model.parameters(), lr=0.001)
            
#             checkpoint = torch.load(model_path, map_location='cpu')
            
#             loaded_keys = set(checkpoint['state_dict'].keys())
#             expected_keys = set(model.state_dict().keys())

#             missing_keys = expected_keys - loaded_keys
#             unexpected_keys = loaded_keys - expected_keys

#             print('='*100)
#             print(f"Missing keys in loaded checkpoint: {missing_keys}")
#             print(f"Unexpected keys in loaded checkpoint: {unexpected_keys}")
#             print('='*100)
            
#             model.load_state_dict(checkpoint['state_dict'])
#             model_optimizer.load_state_dict(checkpoint['optimizer'])
            
#             print('='*100)
#             print("Keys in loaded checkpoint:", checkpoint['state_dict'].keys())
#             print('='*100)
            
#             model.eval()
            
#             src, tgt = process_user_message(user_message)
            
#             with torch.no_grad():
#                 # Generate masks (modify this based on your model's input requirements)
#                 src_mask, tgt_mask = model.generate_mask(src, tgt)

#                 # Adjust this line based on your model's input requirements
#                 response = model(src, tgt)

#             response = {'reply': response}  # Modify this to format the AI response as needed
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             response = {'reply': f'Error: {str(e)}'}

#     return JsonResponse(response)




@csrf_exempt
def ai_response(request):
    print('='*100)
    print('torch version:')
    print(torch.__version__)
    print('='*100)

    # Define the key mapping
    key_mapping = {
        'self_attn.W_q': 'attn.W_q',
        'self_attn.W_k': 'attn.W_k',
        'self_attn.W_v': 'attn.W_v',
        'encoder_attention.fc_q': 'encoder_attention.W_q',
        'encoder_attention.fc_k': 'encoder_attention.W_k',
        'encoder_attention.fc_v': 'encoder_attention.W_v',
        'cross_attn.W_q': 'cross_attn.W_q',
        'cross_attn.W_k': 'cross_attn.W_k',
        'cross_attn.W_v': 'cross_attn.W_v',
        'self_attn.norm1.weight': 'norm1.weight',
        'self_attn.norm1.bias': 'norm1.bias',
        'self_attn.norm2.weight': 'norm2.weight',
        'self_attn.norm2.bias': 'norm2.bias',
        'norm3.weight': 'norm3.weight',
        'norm3.bias': 'norm3.bias',
        'positional_encoding.pe': 'pe',

        
        # Decoder layer 0
        'decoder_layers.0.self_attention.fc_q.bias': 'decoder_layers.0.self_attn.W_q.bias',
        'decoder_layers.0.self_attention.fc_q.weight': 'decoder_layers.0.self_attn.W_q.weight',
        'decoder_layers.0.self_attention.fc_k.weight': 'decoder_layers.0.self_attn.W_k.weight',
        'decoder_layers.0.self_attention.fc_k.bias': 'decoder_layers.0.self_attn.W_k.bias',
        'decoder_layers.0.self_attention.fc_v.weight': 'decoder_layers.0.self_attn.W_v.weight',
        'decoder_layers.0.self_attention.fc_v.bias': 'decoder_layers.0.self_attn.W_v.bias',
        'decoder_layers.0.self_attention.fc_o.weight': 'decoder_layers.0.self_attn.W_o.weight',
        'decoder_layers.0.self_attention.fc_o.bias': 'decoder_layers.0.self_attn.W_o.bias',
        


        # Decoder layer 1
        'decoder_layers.1.self_attention.fc_q.weight': 'decoder_layers.1.self_attn.W_q.weight',
        'decoder_layers.1.self_attention.fc_q.bias': 'decoder_layers.1.self_attn.W_q.bias',
        'decoder_layers.1.self_attention.fc_k.weight': 'decoder_layers.1.self_attn.W_k.weight',
        'decoder_layers.1.self_attention.fc_k.bias': 'decoder_layers.1.self_attn.W_k.bias',
        'decoder_layers.1.self_attention.fc_v.weight': 'decoder_layers.1.self_attn.W_v.weight',
        'decoder_layers.1.self_attention.fc_v.bias': 'decoder_layers.1.self_attn.W_v.bias',
        'decoder_layers.1.self_attention.fc_o.weight': 'decoder_layers.1.self_attn.W_o.weight',
        'decoder_layers.1.self_attention.fc_o.bias': 'decoder_layers.1.self_attn.W_o.bias',

        
        # Decoder layer 3
        'decoder_layers.3.self_attention.fc_q.weight': 'decoder_layers.3.self_attn.W_q.weight',
        'decoder_layers.3.self_attention.fc_q.bias': 'decoder_layers.3.self_attn.W_q.bias',
        'decoder_layers.3.self_attention.fc_k.weight': 'decoder_layers.3.self_attn.W_k.weight',
        'decoder_layers.3.self_attention.fc_k.bias': 'decoder_layers.3.self_attn.W_k.bias',
        'decoder_layers.3.self_attention.fc_v.weight': 'decoder_layers.3.self_attn.W_v.weight',
        'decoder_layers.3.self_attention.fc_v.bias': 'decoder_layers.3.self_attn.W_v.bias',
        'decoder_layers.3.self_attention.fc_o.weight': 'decoder_layers.3.self_attn.W_o.weight',
        'decoder_layers.3.self_attention.fc_o.bias': 'decoder_layers.3.self_attn.W_o.bias',

        
        # Decoder layer 4
        'decoder_layers.4.self_attention.fc_q.weight': 'decoder_layers.4.self_attn.W_q.weight',
        'decoder_layers.4.self_attention.fc_q.bias': 'decoder_layers.4.self_attn.W_q.bias',
        'decoder_layers.4.self_attention.fc_k.weight': 'decoder_layers.4.self_attn.W_k.weight',
        'decoder_layers.4.self_attention.fc_k.bias': 'decoder_layers.4.self_attn.W_k.bias',
        'decoder_layers.4.self_attention.fc_v.weight': 'decoder_layers.4.self_attn.W_v.weight',
        'decoder_layers.4.self_attention.fc_v.bias': 'decoder_layers.4.self_attn.W_v.bias',
        'decoder_layers.4.self_attention.fc_o.weight': 'decoder_layers.4.self_attn.W_o.weight',
        'decoder_layers.4.self_attention.fc_o.bias': 'decoder_layers.4.self_attn.W_o.bias',
        'decoder_layers.4.encoder_attention.fc_q.weight': 'decoder_layers.4.encoder_attn.W_q.weight',
        'decoder_layers.4.encoder_attention.fc_q.bias': 'decoder_layers.4.encoder_attn.W_q.bias',
        'decoder_layers.4.encoder_attention.fc_k.weight': 'decoder_layers.4.encoder_attn.W_k.weight',
        'decoder_layers.4.encoder_attention.fc_k.bias': 'decoder_layers.4.encoder_attn.W_k.bias',
        'decoder_layers.4.encoder_attention.fc_v.weight': 'decoder_layers.4.encoder_attn.W_v.weight',
        'decoder_layers.4.encoder_attention.fc_v.bias': 'decoder_layers.4.encoder_attn.W_v.bias',
        'decoder_layers.4.encoder_attention.fc_o.weight': 'decoder_layers.4.encoder_attn.W_o.weight',
        'decoder_layers.4.encoder_attention.fc_o.bias': 'decoder_layers.4.encoder_attn.W_o.bias',
        'decoder_layers.4.norm1.weight': 'decoder_layers.4.norm1.weight',
        'decoder_layers.4.norm1.bias': 'decoder_layers.4.norm1.bias',
        'decoder_layers.4.norm2.weight': 'decoder_layers.4.norm2.weight',
        'decoder_layers.4.norm2.bias': 'decoder_layers.4.norm2.bias',
        'decoder_layers.4.norm3.weight': 'decoder_layers.4.norm3.weight',
        'decoder_layers.4.norm3.bias': 'decoder_layers.4.norm3.bias',

        
        # Decoder layer 5
        'decoder_layers.5.self_attention.fc_q.weight': 'decoder_layers.5.self_attn.W_q.weight',
        'decoder_layers.5.self_attention.fc_q.bias': 'decoder_layers.5.self_attn.W_q.bias',
        'decoder_layers.5.self_attention.fc_k.weight': 'decoder_layers.5.self_attn.W_k.weight',
        'decoder_layers.5.self_attention.fc_k.bias': 'decoder_layers.5.self_attn.W_k.bias',
        'decoder_layers.5.self_attention.fc_v.weight': 'decoder_layers.5.self_attn.W_v.weight',
        'decoder_layers.5.self_attention.fc_v.bias': 'decoder_layers.5.self_attn.W_v.bias',
        'decoder_layers.5.self_attention.fc_o.weight': 'decoder_layers.5.self_attn.W_o.weight',
        'decoder_layers.5.self_attention.fc_o.bias': 'decoder_layers.5.self_attn.W_o.bias',
        'decoder_layers.5.norm1.weight': 'decoder_layers.5.norm1.weight',
        'decoder_layers.5.norm1.bias': 'decoder_layers.5.norm1.bias',
        'decoder_layers.5.norm2.weight': 'decoder_layers.5.norm2.weight',
        'decoder_layers.5.norm2.bias': 'decoder_layers.5.norm2.bias',
        'decoder_layers.5.norm3.weight': 'decoder_layers.5.norm3.weight',
        'decoder_layers.5.norm3.bias': 'decoder_layers.5.norm3.bias',


        # Decoder layer 6
        'decoder_layers.6.self_attention.fc_q.weight': 'decoder_layers.6.self_attn.W_q.weight',
        'decoder_layers.6.self_attention.fc_q.bias': 'decoder_layers.6.self_attn.W_q.bias',
        'decoder_layers.6.self_attention.fc_k.weight': 'decoder_layers.6.self_attn.W_k.weight',
        'decoder_layers.6.self_attention.fc_k.bias': 'decoder_layers.6.self_attn.W_k.bias',
        'decoder_layers.6.self_attention.fc_v.weight': 'decoder_layers.6.self_attn.W_v.weight',
        'decoder_layers.6.self_attention.fc_v.bias': 'decoder_layers.6.self_attn.W_v.bias',
        'decoder_layers.6.self_attention.fc_o.weight': 'decoder_layers.6.self_attn.W_o.weight',
        'decoder_layers.6.self_attention.fc_o.bias': 'decoder_layers.6.self_attn.W_o.bias',
        
        # Encoder layer 1
        'encoder_layers.1.self_attention.fc_q.weight': 'encoder_layers.1.self_attn.W_q.weight',
        'encoder_layers.1.self_attention.fc_q.bias': 'encoder_layers.1.self_attn.W_q.bias',
        'encoder_layers.1.self_attention.fc_k.weight': 'encoder_layers.1.self_attn.W_k.weight',
        'encoder_layers.1.self_attention.fc_k.bias': 'encoder_layers.1.self_attn.W_k.bias',
        'encoder_layers.1.self_attention.fc_v.weight': 'encoder_layers.1.self_attn.W_v.weight',
        'encoder_layers.1.self_attention.fc_v.bias': 'encoder_layers.1.self_attn.W_v.bias',
        'encoder_layers.1.self_attention.fc_o.weight': 'encoder_layers.1.self_attn.W_o.weight',
        'encoder_layers.1.self_attention.fc_o.bias': 'encoder_layers.1.self_attn.W_o.bias',
        'encoder_layers.1.self_attn.norm1.weight': 'encoder_layers.1.norm1.weight',
        'encoder_layers.1.self_attn.norm1.bias': 'encoder_layers.1.norm1.bias',
        'encoder_layers.1.self_attn.norm2.weight': 'encoder_layers.1.norm2.weight',
        'encoder_layers.1.self_attn.norm2.bias': 'encoder_layers.1.norm2.bias',
        'encoder_layers.1.norm3.weight': 'encoder_layers.1.norm3.weight',
        'encoder_layers.1.norm3.bias': 'encoder_layers.1.norm3.bias',


        # Encoder layer 2
        'encoder_layers.2.self_attention.fc_q.weight': 'encoder_layers.2.self_attn.W_q.weight',
        'encoder_layers.2.self_attention.fc_q.bias': 'encoder_layers.2.self_attn.W_q.bias',
        'encoder_layers.2.self_attention.fc_k.weight': 'encoder_layers.2.self_attn.W_k.weight',
        'encoder_layers.2.self_attention.fc_k.bias': 'encoder_layers.2.self_attn.W_k.bias',
        'encoder_layers.2.self_attention.fc_v.weight': 'encoder_layers.2.self_attn.W_v.weight',
        'encoder_layers.2.self_attention.fc_v.bias': 'encoder_layers.2.self_attn.W_v.bias',
        'encoder_layers.2.self_attention.fc_o.weight': 'encoder_layers.2.self_attn.W_o.weight',
        'encoder_layers.2.self_attention.fc_o.bias': 'encoder_layers.2.self_attn.W_o.bias',
        'encoder_layers.2.self_attn.norm1.weight': 'encoder_layers.2.norm1.weight',
        'encoder_layers.2.self_attn.norm1.bias': 'encoder_layers.2.norm1.bias',
        'encoder_layers.2.self_attn.norm2.weight': 'encoder_layers.2.norm2.weight',
        'encoder_layers.2.self_attn.norm2.bias': 'encoder_layers.2.norm2.bias',
        'encoder_layers.2.norm3.weight': 'encoder_layers.2.norm3.weight',
        'encoder_layers.2.norm3.bias': 'encoder_layers.2.norm3.bias',


        # Encoder layer 3
        'encoder_layers.3.self_attention.fc_q.weight': 'encoder_layers.3.self_attn.W_q.weight',
        'encoder_layers.3.self_attention.fc_q.bias': 'encoder_layers.3.self_attn.W_q.bias',
        'encoder_layers.3.self_attention.fc_k.weight': 'encoder_layers.3.self_attn.W_k.weight',
        'encoder_layers.3.self_attention.fc_k.bias': 'encoder_layers.3.self_attn.W_k.bias',
        'encoder_layers.3.self_attention.fc_v.weight': 'encoder_layers.3.self_attn.W_v.weight',
        'encoder_layers.3.self_attention.fc_v.bias': 'encoder_layers.3.self_attn.W_v.bias',
        'encoder_layers.3.self_attention.fc_o.weight': 'encoder_layers.3.self_attn.W_o.weight',
        'encoder_layers.3.self_attention.fc_o.bias': 'encoder_layers.3.self_attn.W_o.bias',
        'encoder_layers.3.self_attn.norm1.weight': 'encoder_layers.3.norm1.weight',
        'encoder_layers.3.self_attn.norm1.bias': 'encoder_layers.3.norm1.bias',
        'encoder_layers.3.self_attn.norm2.weight': 'encoder_layers.3.norm2.weight',
        'encoder_layers.3.self_attn.norm2.bias': 'encoder_layers.3.norm2.bias',
        'encoder_layers.3.norm3.weight': 'encoder_layers.3.norm3.weight',
        'encoder_layers.3.norm3.bias': 'encoder_layers.3.norm3.bias',

        # Encoder layer 4
        'encoder_layers.4.self_attention.fc_q.weight': 'encoder_layers.4.self_attn.W_q.weight',
        'encoder_layers.4.self_attention.fc_q.bias': 'encoder_layers.4.self_attn.W_q.bias',
        'encoder_layers.4.self_attention.fc_k.weight': 'encoder_layers.4.self_attn.W_k.weight',
        'encoder_layers.4.self_attention.fc_k.bias': 'encoder_layers.4.self_attn.W_k.bias',
        'encoder_layers.4.self_attention.fc_v.weight': 'encoder_layers.4.self_attn.W_v.weight',
        'encoder_layers.4.self_attention.fc_v.bias': 'encoder_layers.4.self_attn.W_v.bias',
        'encoder_layers.4.self_attention.fc_o.weight': 'encoder_layers.4.self_attn.W_o.weight',
        'encoder_layers.4.self_attention.fc_o.bias': 'encoder_layers.4.self_attn.W_o.bias',
        'encoder_layers.4.self_attn.norm1.weight': 'encoder_layers.4.norm1.weight',
        'encoder_layers.4.self_attn.norm1.bias': 'encoder_layers.4.norm1.bias',
        'encoder_layers.4.self_attn.norm2.weight': 'encoder_layers.4.norm2.weight',
        'encoder_layers.4.self_attn.norm2.bias': 'encoder_layers.4.norm2.bias',
        'encoder_layers.4.norm3.weight': 'encoder_layers.4.norm3.weight',
        'encoder_layers.4.norm3.bias': 'encoder_layers.4.norm3.bias',

        # Encoder layer 5
        'encoder_layers.5.self_attention.fc_q.weight': 'encoder_layers.5.self_attn.W_q.weight',
        'encoder_layers.5.self_attention.fc_q.bias': 'encoder_layers.5.self_attn.W_q.bias',
        'encoder_layers.5.self_attention.fc_k.weight': 'encoder_layers.5.self_attn.W_k.weight',
        'encoder_layers.5.self_attention.fc_k.bias': 'encoder_layers.5.self_attn.W_k.bias',
        'encoder_layers.5.self_attention.fc_v.weight': 'encoder_layers.5.self_attn.W_v.weight',
        'encoder_layers.5.self_attention.fc_v.bias': 'encoder_layers.5.self_attn.W_v.bias',
        'encoder_layers.5.self_attention.fc_o.weight': 'encoder_layers.5.self_attn.W_o.weight',
        'encoder_layers.5.self_attention.fc_o.bias': 'encoder_layers.5.self_attn.W_o.bias',
        'encoder_layers.5.self_attn.norm1.weight': 'encoder_layers.5.norm1.weight',
        'encoder_layers.5.self_attn.norm1.bias': 'encoder_layers.5.norm1.bias',
        'encoder_layers.5.self_attn.norm2.weight': 'encoder_layers.5.norm2.weight',
        'encoder_layers.5.self_attn.norm2.bias': 'encoder_layers.5.norm2.bias',
        'encoder_layers.5.norm3.weight': 'encoder_layers.5.norm3.weight',
        'encoder_layers.5.norm3.bias': 'encoder_layers.5.norm3.bias',
        
        
        
        
        'decoder_layers.0.encoder_attention.fc_o.bias': 'decoder_layers.0.self_attention.fc_o.bias',
        'decoder_layers.2.encoder_attention.fc_q.weight': 'decoder_layers.2.self_attention.fc_q.weight',
        'encoder_layers.1.self_attention.fc_q.bias': 'encoder_layers.1.self_attention.fc_q.bias',
        'decoder_layers.3.encoder_attention.fc_v.weight': 'decoder_layers.3.self_attention.fc_v.weight',
        'decoder_layers.4.encoder_attention.fc_k.bias': 'decoder_layers.4.self_attention.fc_k.bias',
        'encoder_layers.5.self_attention.fc_k.bias': 'encoder_layers.5.self_attention.fc_k.bias',
        'decoder_layers.0.encoder_attention.fc_v.bias': 'decoder_layers.0.self_attention.fc_v.bias',
        'decoder_layers.0.self_attention.fc_o.bias': 'decoder_layers.0.self_attention.fc_o.bias',
        'encoder_layers.5.self_attention.fc_q.weight': 'encoder_layers.5.self_attention.fc_q.weight',
        'decoder_layers.0.self_attention.fc_k.weight': 'decoder_layers.0.self_attention.fc_k.weight',
        'encoder_layers.4.self_attention.fc_q.weight': 'encoder_layers.4.self_attention.fc_q.weight',
        'decoder_layers.2.encoder_attention.fc_o.weight': 'decoder_layers.2.self_attention.fc_o.weight',
        'encoder_layers.4.self_attention.fc_o.bias': 'encoder_layers.4.self_attention.fc_o.bias',
        'encoder_layers.1.self_attention.fc_q.weight': 'encoder_layers.1.self_attention.fc_q.weight',
        'decoder_layers.5.encoder_attention.fc_k.bias': 'decoder_layers.5.self_attention.fc_k.bias',
        'decoder_layers.2.self_attention.fc_k.bias': 'decoder_layers.2.self_attention.fc_k.bias',
        'encoder_layers.4.self_attention.fc_q.bias': 'encoder_layers.4.self_attention.fc_q.bias',
        'decoder_layers.1.self_attention.fc_q.bias': 'decoder_layers.1.self_attention.fc_q.bias',
        'encoder_layers.1.self_attention.fc_k.weight': 'encoder_layers.1.self_attention.fc_k.weight',
        'decoder_layers.2.encoder_attention.fc_v.weight': 'decoder_layers.2.self_attention.fc_v.weight',
        'decoder_layers.1.encoder_attention.fc_q.weight': 'decoder_layers.1.self_attention.fc_q.weight',
        'decoder_layers.1.encoder_attention.fc_o.bias': 'decoder_layers.1.self_attention.fc_o.bias',
        'decoder_layers.4.encoder_attention.fc_q.bias': 'decoder_layers.4.self_attention.fc_q.bias',
        'decoder_layers.3.self_attention.fc_k.bias': 'decoder_layers.3.self_attention.fc_k.bias',
        'decoder_layers.2.self_attention.fc_q.weight': 'decoder_layers.2.self_attention.fc_q.weight',
        'decoder_layers.5.self_attention.fc_v.weight': 'decoder_layers.5.self_attention.fc_v.weight',
        'encoder_layers.2.self_attention.fc_q.weight': 'encoder_layers.2.self_attention.fc_q.weight',
        'encoder_layers.4.self_attention.fc_k.bias': 'encoder_layers.4.self_attention.fc_k.bias',
        'decoder_layers.4.self_attention.fc_k.weight': 'decoder_layers.4.self_attention.fc_k.weight',
        'decoder_layers.4.self_attention.fc_v.bias': 'decoder_layers.4.self_attention.fc_v.bias',
        'decoder_layers.3.self_attention.fc_v.bias': 'decoder_layers.3.self_attention.fc_v.bias',
        'encoder_layers.3.self_attention.fc_o.weight': 'encoder_layers.3.self_attention.fc_o.weight',
        'decoder_layers.1.encoder_attention.fc_k.weight': 'decoder_layers.1.self_attention.fc_k.weight',
        'encoder_layers.1.self_attention.fc_v.bias': 'encoder_layers.1.self_attention.fc_v.bias',
        'encoder_layers.4.self_attention.fc_k.weight': 'encoder_layers.4.self_attention.fc_k.weight',
        'decoder_layers.2.encoder_attention.fc_v.bias': 'decoder_layers.2.self_attention.fc_v.bias',
        'encoder_layers.1.self_attention.fc_o.weight': 'encoder_layers.1.self_attention.fc_o.weight',
        'decoder_layers.5.self_attention.fc_q.weight': 'decoder_layers.5.self_attention.fc_q.weight',
        'encoder_layers.0.self_attention.fc_q.bias': 'encoder_layers.0.self_attention.fc_q.bias',
        'decoder_layers.3.self_attention.fc_o.bias': 'decoder_layers.3.self_attention.fc_o.bias',
        'decoder_layers.0.self_attention.fc_q.weight': 'decoder_layers.0.self_attention.fc_q.weight',
        'decoder_layers.3.encoder_attention.fc_k.bias': 'decoder_layers.3.self_attention.fc_k.bias',
        'encoder_layers.5.self_attention.fc_o.bias': 'encoder_layers.5.self_attention.fc_o.bias',
        'decoder_layers.3.encoder_attention.fc_o.bias': 'decoder_layers.3.self_attention.fc_o.bias',
        'decoder_layers.3.self_attention.fc_k.weight': 'decoder_layers.3.self_attention.fc_k.weight',
        'decoder_layers.0.self_attention.fc_o.weight': 'decoder_layers.0.self_attention.fc_o.weight',
        'encoder_layers.2.self_attention.fc_k.bias': 'encoder_layers.2.self_attention.fc_k.bias',
        'encoder_layers.0.self_attention.fc_o.bias': 'encoder_layers.0.self_attention.fc_o.bias',
        'encoder_layers.4.self_attention.fc_v.weight': 'encoder_layers.4.self_attention.fc_v.weight',
        'encoder_layers.0.self_attention.fc_k.bias': 'encoder_layers.0.self_attention.fc_k.bias',
        'decoder_layers.3.encoder_attention.fc_q.bias': 'decoder_layers.3.self_attention.fc_q.bias',
        'decoder_layers.0.encoder_attention.fc_o.weight': 'decoder_layers.0.self_attention.fc_o.weight',
        'decoder_layers.0.self_attention.fc_q.bias': 'decoder_layers.0.self_attention.fc_q.bias',
        'decoder_layers.5.self_attention.fc_o.bias': 'decoder_layers.5.self_attention.fc_o.bias',
        'decoder_layers.1.encoder_attention.fc_o.weight': 'decoder_layers.1.self_attention.fc_o.weight',
        'encoder_layers.2.self_attention.fc_k.weight': 'encoder_layers.2.self_attention.fc_k.weight',
        'decoder_layers.0.self_attention.fc_v.weight': 'decoder_layers.0.self_attention.fc_v.weight',
        'decoder_layers.4.encoder_attention.fc_q.weight': 'decoder_layers.4.self_attention.fc_q.weight',
        'decoder_layers.3.self_attention.fc_v.weight': 'decoder_layers.3.self_attention.fc_v.weight',
        'encoder_layers.5.self_attention.fc_v.bias': 'encoder_layers.5.self_attention.fc_v.bias',
        'encoder_layers.3.self_attention.fc_q.weight': 'encoder_layers.3.self_attention.fc_q.weight',
        'encoder_layers.3.self_attention.fc_q.bias': 'encoder_layers.3.self_attention.fc_q.bias',
        'encoder_layers.0.self_attention.fc_v.bias': 'encoder_layers.0.self_attention.fc_v.bias',
        'decoder_layers.3.self_attention.fc_o.weight': 'decoder_layers.3.self_attention.fc_o.weight',
        'decoder_layers.3.encoder_attention.fc_q.weight': 'decoder_layers.3.self_attention.fc_q.weight',
        'decoder_layers.5.encoder_attention.fc_v.bias': 'decoder_layers.5.self',
        
        'encoder_layers.0.self_attention.fc_q.weight': 'encoder_layers.0.self_attn.W_q.weight',
        'encoder_layers.0.self_attention.fc_q.bias': 'encoder_layers.0.self_attn.W_q.bias',
        'encoder_layers.0.self_attention.fc_k.weight': 'encoder_layers.0.self_attn.W_k.weight',
        'encoder_layers.0.self_attention.fc_k.bias': 'encoder_layers.0.self_attn.W_k.bias',
        'encoder_layers.0.self_attention.fc_v.weight': 'encoder_layers.0.self_attn.W_v.weight',
        'encoder_layers.0.self_attention.fc_v.bias': 'encoder_layers.0.self_attn.W_v.bias',
        'encoder_layers.0.self_attention.fc_o.weight': 'encoder_layers.0.self_attn.W_o.weight',
        'encoder_layers.0.self_attention.fc_o.bias': 'encoder_layers.0.self_attn.W_o.bias',
        'encoder_layers.0.norm1.weight': 'encoder_layers.0.norm1.weight',
        'encoder_layers.0.norm1.bias': 'encoder_layers.0.norm1.bias',
        'encoder_layers.0.norm2.weight': 'encoder_layers.0.norm2.weight',
        'encoder_layers.0.norm2.bias': 'encoder_layers.0.norm2.bias',

        'decoder_layers.0.self_attention.fc_q.weight': 'decoder_layers.0.self_attn.W_q.weight',
        'decoder_layers.0.self_attention.fc_q.bias': 'decoder_layers.0.self_attn.W_q.bias',
        'decoder_layers.0.self_attention.fc_k.weight': 'decoder_layers.0.self_attn.W_k.weight',
        'decoder_layers.0.self_attention.fc_k.bias': 'decoder_layers.0.self_attn.W_k.bias',
        'decoder_layers.0.self_attention.fc_v.weight': 'decoder_layers.0.self_attn.W_v.weight',
        'decoder_layers.0.self_attention.fc_v.bias': 'decoder_layers.0.self_attn.W_v.bias',
        'decoder_layers.0.self_attention.fc_o.weight': 'decoder_layers.0.self_attn.W_o.weight',
        'decoder_layers.0.self_attention.fc_o.bias': 'decoder_layers.0.self_attn.W_o.bias',
        'decoder_layers.0.encoder_attention.fc_q.weight': 'decoder_layers.0.encoder_attn.W_q.weight',
        'decoder_layers.0.encoder_attention.fc_q.bias': 'decoder_layers.0.encoder_attn.W_q.bias',
        'decoder_layers.0.encoder_attention.fc_k.weight': 'decoder_layers.0.encoder_attn.W_k.weight',
        'decoder_layers.0.encoder_attention.fc_k.bias': 'decoder_layers.0.encoder_attn.W_k.bias',
        'decoder_layers.0.encoder_attention.fc_v.weight': 'decoder_layers.0.encoder_attn.W_v.weight',
        'decoder_layers.0.encoder_attention.fc_v.bias': 'decoder_layers.0.encoder_attn.W_v.bias',
        'decoder_layers.0.encoder_attention.fc_o.weight': 'decoder_layers.0.encoder_attn.W_o.weight',
        'decoder_layers.0.encoder_attention.fc_o.bias': 'decoder_layers.0.encoder_attn.W_o.bias',
        'decoder_layers.0.norm1.weight': 'decoder_layers.0.norm1.weight',
        'decoder_layers.0.norm1.bias': 'decoder_layers.0.norm1.bias',
        'decoder_layers.0.norm2.weight': 'decoder_layers.0.norm2.weight',
        'decoder_layers.0.norm2.bias': 'decoder_layers.0.norm2.bias',
        'decoder_layers.0.norm3.weight': 'decoder_layers.0.norm3.weight',
        'decoder_layers.0.norm3.bias': 'decoder_layers.0.norm3.bias',

    }

    user_message = request.POST.get('userMessage', '')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insura_transformer_model.pth')

    if not os.path.exists(model_path):
        response = {'reply': 'No Model found in the directory'}
    else:
        try:
            # Create an instance of your model class after loading the checkpoint
            model = InsuraTransformer(src_vocab_size=10000, tgt_vocab_size=5000, d_model=512,
                                       num_heads=8, num_layers=6, d_ff=2048, max_seq_length=50, dropout=0.1)

            model_optimizer = optim.Adam(model.parameters(), lr=0.001)

            checkpoint = torch.load(model_path, map_location='cpu')

            # Apply the key mapping
            checkpoint['state_dict'] = {key_mapping.get(k, k): v for k, v in checkpoint['state_dict'].items()}

            loaded_keys = set(checkpoint['state_dict'].keys())
            expected_keys = set(model.state_dict().keys())

            missing_keys = expected_keys - loaded_keys
            unexpected_keys = loaded_keys - expected_keys

            print('=' * 100)
            print(f"Missing keys in loaded checkpoint: {missing_keys}")
            print(f"Unexpected keys in loaded checkpoint: {unexpected_keys}")
            print('=' * 100)

            model.load_state_dict(checkpoint['state_dict'])
            model_optimizer.load_state_dict(checkpoint['optimizer'])

            print('=' * 100)
            print("Keys in loaded checkpoint:", checkpoint['state_dict'].keys())
            print('=' * 100)

            model.eval()

            src, tgt = process_user_message(user_message)

            with torch.no_grad():
                # Generate masks (modify this based on your model's input requirements)
                src_mask, tgt_mask = model.generate_mask(src, tgt)

                # Adjust this line based on your model's input requirements
                response = model(src, tgt)

            response = {'reply': response}  # Modify this to format the AI response as needed
        except Exception as e:
            import traceback
            traceback.print_exc()
            response = {'reply': f'Error: {str(e)}'}

    return JsonResponse(response)

