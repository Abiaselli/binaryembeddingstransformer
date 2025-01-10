import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox, StringVar, OptionMenu
import os
import threading
import logging
import json

def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")


device = "cuda" if torch.cuda.is_available() else "cpu"
# Define EOS token (outside the standard byte range)
EOS= "ÿÿÿÿ"
EOS_TOKEN = "11111111 11111111 11111111 11111111" # 32 digits
EOS_BINARY = 0o11111111111111111111111111111111 # 32 digits
EOS_BINARY_INT = int("11111111111111111111111111111111", 2)  # Converts binary to int
EOS_Twos_complement = "0000000000000000000000000000000011111111111111111111111111111111" #64 digits
EOS_HEX = "FFFFFFFF" #10 digits
EOS_INT = 4294967295
if EOS_BINARY_INT > 2**31 - 1:  # Ensure it doesn't exceed the range for a 32-bit int
    EOS_BINARY_INT = EOS_BINARY_INT % (2**31)
EOS_BINARY_FLOAT = float(EOS_BINARY_INT)
token_length = 32 #must match length of EOS binary
max_seq_len = 64

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(os.environ.get("CUDA_LAUNCH_BLOCKING"))



# RMS Normalization Function
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        r = 1 / torch.sqrt(torch.clamp(variance + eps, min=1e-10))  # Prevent division by zero
        y = r * (x - mean)
        ctx.save_for_backward(x, mean, variance, r)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, variance, r = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]
        denom = variance + eps
        denom = torch.clamp(denom, min=1e-8)  # Ensure denom is not too small
        grad_input = (1 / N) * r * (
            N * grad_output
            - grad_output.sum(dim=-1, keepdim=True)
            - (x - mean) * ((grad_output * (x - mean)).sum(dim=-1, keepdim=True) / denom)
        )
        return grad_input, None

def prepare_src_mask(mask, batch_size, num_heads):
    # Expand the mask to [batch_size * num_heads, seq_len, seq_len]
    mask = mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
    return mask

def custom_collate_fn(batch):
        """
        batch is a list of (query, target) pairs
        we want to return two lists (or Tensors) of queries and targets
        """
        queries, targets = zip(*batch)  # unzip into separate tuples
        # queries and targets are now tuples of length = batch_size
        
        # Return them as lists or keep them as tuples
        return list(queries), list(targets)
    
def rms_norm(x, eps=1e-8):
    return RMSNormFunction.apply(x, eps)

# Activation quantization function
def activation_quant(x, bits=8):
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    x_abs_max = x.abs().max()
    if x_abs_max == 0 or torch.isnan(x_abs_max):
        scale = 1.0  # Avoid division by zero
    else:
        scale = x_abs_max / qmax
    x_quant = torch.clamp((x / scale).round(), qmin, qmax)
    x_dequant = x_quant * scale
    return x_dequant

# Custom Ternary Weight Function
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, weight):
        # Ternarize weights to -1, 0, or +1
        ternary_weight = torch.sign(weight)
        return ternary_weight

    @staticmethod
    def backward(_ctx, grad_output):
        # Gradient is passed through unchanged
        grad_input = grad_output.clone()
        return grad_input

def ternarize_weight(weight):
    return TernaryWeightFunction.apply(weight)

# Matmul-free linear function with quantization
def matmul_free_linear(input, weight):
    # Quantize input and weight
    input_q = activation_quant(input)
    weight_q = ternarize_weight(weight)
    logging.debug(f"input_q shape: {input_q.shape}, weight_q shape: {weight_q.shape}")

    # Perform matrix multiplication
    output = input_q.matmul(weight_q.t())
    return output



def preprocess_text(text, max_seq_len=max_seq_len, chunk_size=token_length):
    binary_sequence = []
    for char in text:
        char_binary = format(ord(char), '08b')
        binary_sequence.extend([int(bit) for bit in char_binary])

    eos_binary = [int(bit) for bit in EOS_TOKEN.replace(" ", "")]
    max_binary_length = max_seq_len * chunk_size

    # Truncate and add EOS token
    binary_sequence = binary_sequence[:max_binary_length - len(eos_binary)]
    binary_sequence.extend(eos_binary)

    # Pad to `max_binary_length`
    padding_needed = max_binary_length - len(binary_sequence)
    binary_sequence.extend([0] * padding_needed)

    # Chunk the sequence
    chunks = [binary_sequence[i:i + chunk_size] for i in range(0, max_binary_length, chunk_size)]
    return torch.tensor(chunks, dtype=torch.float32)  # Shape: [seq_len, chunk_size]

def preprocess_batch(text_batch, max_seq_len=max_seq_len, chunk_size=token_length):
    """
    Preprocess a batch of strings into binary tensors, ensuring consistent padding.
    """
    processed_batch = [preprocess_text(text, max_seq_len, chunk_size) for text in text_batch]
    
    # Stack tensors, ensuring all are [seq_len, chunk_size]
    return torch.stack(processed_batch, dim=0)  # Shape: [batch_size, seq_len, chunk_size]



def bytes_to_embeddings(byte_sequences, embed_size, device, binary_length=token_length):
    """
    Converts byte sequences into binary-based embeddings, handling batched inputs.
    Expects `byte_sequences` as a tensor of shape [batch_size, seq_len, binary_len].
    """    
    if not isinstance(byte_sequences, torch.Tensor):
        raise ValueError("Input to bytes_to_embeddings must be a torch.Tensor")

    batch_size, seq_len, bin_len = byte_sequences.shape
    if bin_len != binary_length:
        raise ValueError(f"Expected binary length {binary_length}, got {bin_len}")

    binary_embedding = BinaryEmbedding(embed_size).to(device)
    processed_embeddings, probabilities_list = [], []
    logging.debug(f"Initializing BinaryEmbedding with embed_size: {embed_size}")

    for batch_idx in range(batch_size):
        sequence = byte_sequences[batch_idx]  # Shape: [seq_len, binary_len]

        try:
            # Validate sequence shape
            if sequence.shape[-1] != binary_length:
                raise ValueError(f"Binary length mismatch: {sequence.shape[-1]} != {binary_length}")
            # Stack binary chunks
            #padded_tensor = torch.stack(binary_chunks).to(device ) # Shape: [seq_len, binary_len]
            #logging.debug(f"Padded tensor shape: {padded_tensor.shape}")

            embedding_tensor, probabilities = binary_embedding(sequence)
            logging.debug(f"BTWbinary embedding shape: {embedding_tensor.shape}")
            logging.debug(f"BTWProbabilities shape: {probabilities.shape}")
            processed_embeddings.append(embedding_tensor)
            probabilities_list.append(probabilities)

        except Exception as e:
            logging.error(f"Error processing sequence {sequence}: {e}")
            raise
        
    logging.debug(f"Processed embedding list length: {len(processed_embeddings)}")
    logging.debug(f"Probabilities list length: {len(probabilities_list)}")
    return torch.stack(processed_embeddings), torch.stack(probabilities_list)


def bytes_to_embeddings_single(query_tensor, embed_size, device, binary_length=token_length):
    """
    Converts a single query tensor into binary-based embeddings.
    """
    try:
        # Split query tensor into binary chunks
        binary_chunks = list(torch.split(query_tensor, binary_length))

        # Ensure all chunks are the correct size
        binary_chunks = [chunk for chunk in binary_chunks if chunk.numel() == binary_length]

        if not binary_chunks:
            raise ValueError("No valid binary chunks found. Check input sequence length.")

        logging.debug(f"Binary chunks: {[chunk.shape for chunk in binary_chunks]}")

        # Stack and process binary chunks
        padded_tensor = torch.stack(binary_chunks).to(device)  # Shape: [seq_len, binary_length]

        # Initialize BinaryEmbedding
        binary_embedding = BinaryEmbedding(embed_size).to(device)
        binary_embedding, probabilities = binary_embedding(padded_tensor.unsqueeze(0))  # Add batch dim temporarily

        logging.debug(f"binary embedding shape: {binary_embedding.shape}")
        logging.debug(f"Probabilities shape: {probabilities.shape}")

        return binary_embedding.squeeze(0), probabilities.squeeze(0)  # Remove batch dim
    except Exception as e:
        logging.error(f"Error processing query tensor: {e}")
        raise


# MatMul-free Linear Gated Recurrent Unit (MLGRU) Cell
class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Weights and biases
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x_t, h_t_minus_1):
        # Apply RMS normalization
        x_t = rms_norm(x_t, self.eps)
        logging.debug(f"x_t shape: {x_t.shape}, W_f shape: {self.W_f.shape}")

        # Linear operations
        f_t_linear = matmul_free_linear(x_t, self.W_f) + self.b_f
        c_t_linear = matmul_free_linear(x_t, self.W_c) + self.b_c
        g_t_linear = matmul_free_linear(x_t, self.W_g) + self.b_g

        # Activation functions
        sig_f_t = torch.sigmoid(f_t_linear)
        silu_c_t = F.silu(c_t_linear)
        sig_g_t = torch.sigmoid(g_t_linear)

        # Hidden state computations
        h_t = sig_f_t * h_t_minus_1 + (1 - sig_f_t) * silu_c_t
        o_t = h_t * sig_g_t

        return o_t, h_t


# MLGRU Layer
class MLGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.cell = MLGRUCell(input_size, hidden_size, eps)
        self.hidden_size = hidden_size

    def forward(self, x):
        logging.debug(f"Shape of x in MLGRULayer: {x.shape}")  

        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            outputs.append(o_t.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output


# MatMul-free GLU
class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))
        self.b_u = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Apply RMS normalization
        x = rms_norm(x, self.eps)
        # Quantize activations
        x = activation_quant(x)

        # Linear operations
        g_t = matmul_free_linear(x, self.W_g) + self.b_g
        u_t = matmul_free_linear(x, self.W_u) + self.b_u

        # Activation functions
        g_t = F.silu(g_t)
        p_t = g_t * u_t  # Assuming linear activation

        # Output layer
        d_t = matmul_free_linear(p_t, self.W_d) + self.b_d

        return d_t

class MiniTransformerNode(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, hidden_size, vocab_size=token_length, max_seq_length=max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(hidden_size, embed_size)
        max_positions = max_seq_length*token_length
        self.pos_encoder = nn.Embedding(max_positions, embed_size)
        self.num_heads=num_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size=hidden_size
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.device=device


    def forward(self, x, prev_node_output=None, src_mask=None, is_final_node=False):
        logging.debug(f"Input to mini transformer node shape: {x.shape}")
        #input shaoe [batch_size, seq_len*binary_len, embed_size (d/e)]
        embeddings = torch.einsum("bsd,de->bse", x, self.embedding.weight)
        logging.debug(f"Input to MiniTransformerNode shape: {embeddings.shape}")
        batch_size = embeddings.size(0)
        seq_length = embeddings.size(1)
        binary_len = token_length
        positions = torch.arange(seq_length, device=x.device)
        logging.debug(f"Positions shape: {positions.shape}")

        logging.debug(f"Positions Tensor: {positions}")
        logging.debug(f"Max Position Index: {positions.max()}, Min Position Index: {positions.min()}")

        logging.debug(f"Position Encoder Weight Size: {self.pos_encoder.weight.size()}")
        assert positions.max() < self.pos_encoder.weight.size(0), (
            f"Invalid position index {positions.max()}, exceeds embedding size {self.pos_encoder.weight.size(0)}"
        )

        pos_encodings = self.pos_encoder(positions)
        logging.debug(f"Positional encodings shape: {pos_encodings.shape}")
        logging.debug(f"Positional encodings (sample): {pos_encodings[:5]}")
        # Add positional encodings to embeddings
        logging.debug(f"Embeddings shape for pos: {embeddings.shape}")
        logging.debug(f"POS_encodings shape for pos: {pos_encodings.shape}")

        src = embeddings + pos_encodings
        logging.debug(f"SRC shape for pos: {src.shape}")

        
        src = embeddings + pos_encodings  # [seq_len, embed_size] or [batch_size, seq_len, embed_size]
        seq_len = src.size(1) if src.dim() == 3 else src.size(0)
        num_heads = self.num_heads

        # Generate the attention mask
        src_mask = generate_attention_mask(embeddings, num_heads, seq_len)

        logging.debug(f"SRC mask shape: {src_mask.shape}")

        logging.debug(f"SRC mask shape for transformer encoder: {src_mask.shape}")

        # Pass through the Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        logging.debug(f"Output shape from transformer encoder: {output.shape}")
        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            # Generate a new attention mask
            attn_mask = generate_attention_mask(output, num_heads, seq_len).to(self.device)
            logging.debug(f"Attention mask shape: {attn_mask.shape}")

            if src_mask is not None:
                # Align src_mask to match attn_mask
                seq_length, binary_length =output.size(0), output.size(1)

                # Ensure src_mask is [batch_size, seq_len, seq_len]
                #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
                logging.debug(f"src_mask shape before repeat1: {src_mask.shape}")

                # Combine masks
                attn_mask = attn_mask * src_mask
                logging.debug(f"Final attn_mask shape1: {attn_mask.shape}")
                logging.debug(f"Final src_mask shape1: {src_mask.shape}")

            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output1: {output.shape}")
            logging.debug(f"Shape of attention_weights1: {attention_weights.shape}")
        else:
            attn_mask = torch.zeros_like(src_mask)

            # Ensure src_mask is [batch_size, seq_len, seq_len]
            #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
            logging.debug(f"src_mask shape before repeat2: {src_mask.shape}")

            # Combine masks
            attn_mask = attn_mask * src_mask
            logging.debug(f"Final attn_mask shape2: {attn_mask.shape}")
            logging.debug(f"Final src_mask shape2: {src_mask.shape}")
            prev_node_output = torch.zeros_like(output)
            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output2: {output.shape}")
            logging.debug(f"Shape of attention_weights2: {attention_weights.shape}")

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output
            logging.debug(f"Shape of concatenated output: {output.shape}")

        if is_final_node:
            logging.debug(f"Shape of final bias1: {self.fc_out.bias.shape}")

            logging.debug(f"Shape of output for bias final node: {output.shape}")

            output = self.fc_out(output) + self.fc_out.bias
            logging.debug(f"Shape of final output1: {output.shape}")
            logging.debug(f"Shape of final attention_weight1s: {attention_weights.shape}")  
        else:
            # [batch_size, seq_len*binary_len, embed_size] and [binary_len]
            logging.debug(f"Shape of final bias2: {self.fc_out.bias.shape}")
            # Reshape and expand bias
            bias = self.fc_out.bias.view(1, binary_len, 1)  # Shape: [1, binary_len, 1]
            logging.debug(f"Bias shape before expansion: {bias.shape}")  # Expected: [binary_len]

            # Step 2: Repeat to expand middle dimension to match seq_len
            repeat_factor = seq_len // binary_len  # Calculate how many times to repeat
            bias_expanded = bias.repeat(1, repeat_factor, 1)  # Expand to [1, seq_len, embed_size]

            logging.debug(f"Bias shape after repeat: {bias_expanded.shape}")
            bias_expanded = bias_expanded.expand(batch_size, seq_len, self.hidden_size)  # Shape: [batch_size, seq_len, embed_size]

            logging.debug(f"Shape of bias_expanded: {bias_expanded.shape}")
            logging.debug(f"Shape of output for fc_out.weight: {output.shape}")
            logging.debug(f"Shape of fc_out.weight: {self.fc_out.weight.shape}")
            fc_out_expanded = self.fc_out.weight.repeat(1, repeat_factor, 1)  # Expand to [1, seq_len, embed_size]
            logging.debug(f"Shape of fc_out_expanded: {fc_out_expanded.shape}")
            output = output + fc_out_expanded
            logging.debug(f"Shape of output for bias: {output.shape}")
            output = output + bias_expanded

            logging.debug(f"Shape of final output2: {output.shape}")
            logging.debug(f"Shape of final attention_weights2: {attention_weights.shape}")
            
        return output, attention_weights


class BinaryCascadeTransformer(nn.Module):
    def __init__(self, num_nodes, hidden_size, num_heads, max_seq_length, vocab_size, node_type, num_layers, include_preprocessing_node=True):
        super(BinaryCascadeTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.node_type = node_type
        self.num_layers=num_layers
        self.embed_size=hidden_size
        self.output_layer = nn.Linear(vocab_size, 2)  # Predict binary probabilities for vocab_size binary positions.
        self.device=device

        if include_preprocessing_node:
            self.preprocessing_node = ModifiedTransformerNode(
                embed_size=hidden_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_length=max_seq_length
            )

        # Initialize nodes based on the selected architecture
        if node_type == "matmul_free":
            self.nodes = nn.ModuleList([
                MatMulFreeLanguageModel(hidden_size, hidden_size, num_heads, max_seq_length)
                for _ in range(num_nodes)
            ])
        elif node_type == "mini_transformer":
            self.nodes = nn.ModuleList([
                MiniTransformerNode(self.embed_size, num_heads, num_layers, hidden_size, vocab_size, max_seq_length)
                for _ in range(num_nodes)
            ])
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def forward(self, input_text, mask=None):

        logging.debug(f"Starting forward pass with input shape: {input_text.shape}")
        binary_embeddings = self.preprocessing_node(input_text)
        logging.debug(f"Binary embeddings shape: {binary_embeddings.shape}")

        # Pass through the transformer layers without flattening
        prev_node_output = None
        attention_weights_all_nodes = []

        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            binary_embeddings, attention_weights = node(
                binary_embeddings, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node
            )
            prev_node_output = binary_embeddings

            attention_weights_all_nodes.append(attention_weights)
        
        binary_embeddings=self.output_layer(binary_embeddings) + self.output_layer.bias
        logging.debug(f"binary embeddings output shape: {binary_embeddings.shape}")
        
        return binary_embeddings, attention_weights_all_nodes


    def target_projection(self, target_input, mask=None):
        """
        Convert a 2D tensor [seq_len, binary_len] of 0/1 bits into [seq_len, binary_len, 1].
        """
        # Make sure target_input is float for loss.
        
        return target_input.unsqueeze(-1)


class BinaryEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(BinaryEmbedding, self).__init__()
        self.hidden_size = embed_size
        self.vocab_size = token_length
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)  # Learnable adjustment

    def forward(self, binary_input):
        try:
            if binary_input.dim() == 3:  # [batch_size, seq_len, binary_len]
                batch_size, seq_len, bin_len = binary_input.shape
                binary_input = binary_input.view(batch_size * seq_len, bin_len)
            else:
                batch_size = 1

            # Validate input dimensions
            assert binary_input.dim() == 2, "binary_input must be 2D (seq_len, binary_length)"

            seq_len, binary_len = binary_input.shape

            # 1. Map [0, 1] -> [-1, +1]
            amplitude = binary_input * 2 - 1  # Shape: [seq_len, binary_len]
            logging.debug(f"Amplitude shape: {amplitude.shape}")
            logging.debug(f"Amplitude: {amplitude[:10]}")
            # 2. Create indices for positions
            position_indices = torch.arange(binary_len, device=binary_input.device).unsqueeze(0).repeat(seq_len, 1)
            logging.debug(f"Position indices shape after arrange: {position_indices.shape}")
            # 3. Generate embeddings
            base_embeddings = self.embeddings(position_indices)
            logging.debug(f"Base Embeddings shape: {base_embeddings.shape}")
            logging.debug(f"Base Embeddings: {base_embeddings[:10]}")            
            amplitude = amplitude.unsqueeze(-1)  # [seq_len, binary_len, 1]
            embeddings = base_embeddings * amplitude
            logging.debug(f"Embeddings shape: {embeddings.shape}")
            logging.debug(f"Embeddings: {embeddings[:10]}") 
            # 4. Sum along `binary_len` dimension for logits
            logit_prime = torch.sum(embeddings, dim=2, keepdim=True)  # Shape: [seq_len, binary_len, 1]
            logging.debug(f"Logit prime shape: {logit_prime.shape}")

            logging.debug(f"Logit prime after sum: {logit_prime[:10]}") 
            # Reshape back for batch processing
            if batch_size > 1:
                embeddings = embeddings.view(batch_size, seq_len, binary_len, self.hidden_size)
                logit_prime = logit_prime.view(batch_size, seq_len, binary_len, 1)
            logging.debug(f"Embeddings shape before return: {embeddings.shape}")

            logging.debug(f"Logit prime shape before return: {logit_prime.shape}")

            return embeddings, logit_prime

        except Exception as e:
            logging.error(f"Error in BinaryEmbedding forward pass: {e}")
            raise

class BinaryEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_length, device =device):
        """
        binaryEmbeddingLayer dynamically generates binary-based embeddings aligned with the model's hidden size.
        
        Args:
            max_seq_length: Maximum sequence length.
            hidden_size: Model's hidden size (output dimension).
            num_heads: Number of attention heads (for compatibility).
            device: Device for computations (CPU or CUDA).
        """
        super(BinaryEmbeddingLayer, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device

    def forward(self, text_batch):
        """
        Converts input sequences (text batch) into binary embeddings.
        """
        for i, t in enumerate(text_batch):
            logging.debug(f"Element {i} type: {type(t)}, shape: {getattr(t, 'shape', None)}")

        if isinstance(text_batch, list):
            # Ensure all elements in the list are tensors
            text_batch = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in text_batch]
            
            # Use stack if all tensors have the same shape
            try:
                text_batch = torch.stack(text_batch, dim=0)
            except RuntimeError as e:
                logging.error(f"Error stacking tensors: {e}. Trying concatenation.")
                # Fallback to concatenation if shapes differ
                text_batch = torch.cat([t.unsqueeze(0) for t in text_batch], dim=0)

        if text_batch.dim() != 3:
            raise ValueError(f"Expected input shape [batch_size, seq_len, binary_len], got {text_batch.shape}")

        embeddings, probabilities = bytes_to_embeddings(
            byte_sequences=text_batch,
            embed_size=self.hidden_size,
            device=self.device
        )
        return embeddings, probabilities


class ModifiedTransformerNode(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, num_layers, max_seq_length):
        super().__init__()
        self.embedding_layer = BinaryEmbeddingLayer(
            max_seq_length=max_seq_length, hidden_size=hidden_size, num_heads=num_heads, device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, embed_size)

    def forward(self, text_batch):
        # Generate embeddings
        embeddings, _ = self.embedding_layer(text_batch)  # Shape: [batch_size, seq_len, binary_len, embed_size]
        logging.debug(f"Embeddings input shape: {embeddings.shape}")

        # Correctly handle the mbeddings shape

        bsz, seq_len, bin_len, embed_size = embeddings.shape

        # Reshape to the expected three dimensions: [batch_size, seq_len*token_length, embed_size]
        # Assuming vocab_size == binary_length == token_length
        embeddings = embeddings.view(bsz, seq_len * bin_len, embed_size)

        logging.debug(f"Adjusted embeddings shape: {embeddings.shape}")


        # Flatten for transformer
        encoded_embeddings = self.transformer_encoder(embeddings)
        # After preprocessing_node
        logging.debug(f"Preprocessed embeddings shape before fc_out: {encoded_embeddings.shape}")
        preprocessed_embeddings = self.fc_out(encoded_embeddings)
        # After preprocessing_node
        logging.debug(f"Preprocessed embeddings shape: {preprocessed_embeddings.shape}")

        return preprocessed_embeddings

def sample_from_probabilities(probabilities, threshold=0.5):
    """
    Sample binary values from probabilities using a threshold.
    """
    return (probabilities >= threshold).int()  # Cast to int for compatibility with decoding


def decode_binary_sequence(binary_sequence):
    """
    Decodes a binary sequence back into text.
    """
    try:
        # Flatten the binary sequence if nested
        if isinstance(binary_sequence[0], list):
            binary_sequence = [bit for sublist in binary_sequence for bit in sublist]

        logging.debug(f"Binary sequence before conversion: {binary_sequence}")

        # Ensure binary values are cast to integers
        binary_sequence = [int(bit) for bit in binary_sequence]
        binary_string = ''.join(map(str, binary_sequence))

        logging.debug(f"Binary string: {binary_string}")

        # Break the binary string into 8-bit bytes
        bytes_array = [binary_string[i:i + 8] for i in range(0, len(binary_string), 8)]

        logging.debug(f"Bytes array: {bytes_array}")

        # Convert bytes to characters
        decoded_text = ''.join(chr(int(byte, 2)) for byte in bytes_array if int(byte, 2) > 0)

        logging.debug(f"Decoded text: {decoded_text}")

        return decoded_text
    except ValueError as e:
        logging.error(f"Error decoding binary sequence: {e}")
        return ""


# MatMul-Free Language Model
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, max_seq_length, vocab_size=token_length, eps=1e-8, device =device):
        super().__init__()
        self.eps = eps
        self.embedding = nn.Embedding(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)  # Predict binary probabilities for vocab_size binary positions.
        self.max_seq_length = max_seq_length
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.hidden_size=hidden_size
        self.device= device

    def forward(self, input_ids, prev_node_output=None, src_mask=None, is_final_node=False):

        logging.debug(f"Shape of input_ids to mmflm:{input_ids.shape}") 

        #input shaoe [batch_size, seq_len*binary_len, embed_size (d/e)]

        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        binary_len = token_length

        # input Shape: (batch_size, seq_length*binary_len, embed_size)
        x = torch.einsum("bsd,de->bse", input_ids, self.embedding.weight)
        logging.debug(f"Input to mini transformer node shape: {x.shape}")
        #output shaoe [batch_size, seq_len*binary_len, hidden_size ]

        

        logging.debug(f"num_heads in MatMulFreeLanguageModel: {self.num_heads}")

        logging.debug(f"Shape of x after embedding:{x.shape}") 
        x = self.mlgru_layer(x)
        logging.debug(f"Shape of x after mlgru_layer:{x.shape}") 
        x = self.glu(x)
        logging.debug(f"Shape of x after glu:{x.shape}") 
        seq_len = x.size(1) if x.dim() == 3 else x.size(0)
        num_heads = self.num_heads
        # Generate the attention mask
        src_mask = generate_attention_mask(x, num_heads, seq_len)
        logging.debug(f"SRC mask shape: {src_mask.shape}")
        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        output = activation_quant(x)
        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            # Generate a new attention mask
            attn_mask = generate_attention_mask(output, num_heads, seq_len).to(self.device)
            logging.debug(f"Attention mask shape: {attn_mask.shape}")

            if src_mask is not None:
                # Align src_mask to match attn_mask
                seq_length, binary_length =output.size(0), output.size(1)

                # Ensure src_mask is [batch_size, seq_len, seq_len]
                #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
                logging.debug(f"src_mask shape before repeat1: {src_mask.shape}")

                # Combine masks
                attn_mask = attn_mask * src_mask
                logging.debug(f"Final attn_mask shape1: {attn_mask.shape}")
                logging.debug(f"Final src_mask shape1: {src_mask.shape}")

            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output1: {output.shape}")
            logging.debug(f"Shape of attention_weights1: {attention_weights.shape}")
        else:
            attn_mask = torch.ones_like(src_mask)

            # Ensure src_mask is [batch_size, seq_len, seq_len]
            #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
            logging.debug(f"src_mask shape before repeat2: {src_mask.shape}")

            # Combine masks
            attn_mask = attn_mask * src_mask
            logging.debug(f"Final attn_mask shape2: {attn_mask.shape}")
            logging.debug(f"Final src_mask shape2: {src_mask.shape}")
            prev_node_output = torch.zeros_like(output)

            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output2: {output.shape}")
            logging.debug(f"Shape of attention_weights2: {attention_weights.shape}")

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output
            logging.debug(f"Shape of concatenated output: {output.shape}")

        if is_final_node:
            logging.debug(f"Shape of final bias1: {self.output_layer.bias.shape}")
            logging.debug(f"Shape of output for bias final node: {output.shape}")
            output = self.output_layer(output) + self.output_layer.bias
            logging.debug(f"Shape of final output1: {output.shape}")
            logging.debug(f"Shape of final attention_weight1s: {attention_weights.shape}")  
        else:
            # [batch_size, seq_len*binary_len, embed_size] and [binary_len]
            logging.debug(f"Shape of final bias2: {self.output_layer.bias.shape}")
            # Reshape and expand bias
            bias = self.output_layer.bias.view(1, binary_len, 1)  # Shape: [1, binary_len, 1]
            logging.debug(f"Bias shape before expansion: {bias.shape}")  # Expected: [binary_len]

            # Step 2: Repeat to expand middle dimension to match seq_len
            repeat_factor = seq_len // binary_len  # Calculate how many times to repeat
            bias_expanded = bias.repeat(1, repeat_factor, 1)  # Expand to [1, seq_len, embed_size]

            logging.debug(f"Bias shape after repeat: {bias_expanded.shape}")
            bias_expanded = bias_expanded.expand(batch_size, seq_len, self.hidden_size)  # Shape: [batch_size, seq_len, embed_size]

            logging.debug(f"Shape of bias_expanded: {bias_expanded.shape}")
            logging.debug(f"Shape of output for fc_out.weight: {output.shape}")
            logging.debug(f"Shape of output_layer.weight: {self.output_layer.weight.shape}")
            fc_out_expanded = self.output_layer.weight.repeat(1, repeat_factor, 1)  # Expand to [1, seq_len, embed_size]
            logging.debug(f"Shape of fc_out_expanded: {fc_out_expanded.shape}")
            output = output + fc_out_expanded
            logging.debug(f"Shape of output for bias: {output.shape}")
            output = output + bias_expanded


            logging.debug(f"Shape of final output2: {output.shape}")
            logging.debug(f"Shape of final attention_weights2: {attention_weights.shape}")
            
        return output, attention_weights

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_attention_mask_original(embeddings, num_heads):
    """
    Generates a valid attention mask for `torch.nn.MultiheadAttention`.
    Args:
        embeddings: Input embeddings tensor of shape [seq_len, vocab_size, embed_size].
        num_heads: Number of attention heads.

    Returns:
        mask: A 3D attention mask of shape [num_heads, seq_len, seq_len].
    """
    logging.debug(f"Embeddings shape before base mask: {embeddings.shape}")

    # Generate a mask [seq_len]
    seq_len = embeddings.size(0)
    base_mask = torch.ones(seq_len, seq_len, device=embeddings.device)  # Allow all attention by default

    # Expand to [num_heads, seq_len, seq_len]
    head_mask = base_mask.unsqueeze(0).expand(num_heads, seq_len, seq_len)

    logging.debug(f"Generated attention mask shape: {head_mask.shape}")
    return head_mask

def generate_attention_mask_deprecated(embeddings, num_heads, seq_len):
    """
    Generate attention mask that aligns with multi-head expectations.
    Mask shape: [num_heads, seq_len, seq_len].
    """
    # Start with a base diagonal mask or allow-all mask
    base_mask = torch.ones(seq_len, seq_len, device=embeddings.device)

    # Expand to [num_heads, seq_len, seq_len]
    expanded_mask = base_mask.unsqueeze(0).expand(num_heads, -1, -1)

    # If batching, adjust further
    batch_size = embeddings.size(0) if embeddings.dim() == 3 else 1
    if batch_size > 1:
        expanded_mask = expanded_mask.unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)
        expanded_mask = expanded_mask.reshape(batch_size * num_heads, seq_len, seq_len)
    
    return expanded_mask


def generate_attention_mask(embeddings, num_heads, seq_len):
    logging.debug(f"Embeddings for attn mask shape: {embeddings.shape}")
    batch_size, flattened_seq_len, embed_size = embeddings.size()
    binary_len = flattened_seq_len // seq_len
    logging.debug(f"Batch size: {batch_size}, Seq len: {seq_len}, Embed size: {embed_size}")

    if flattened_seq_len != seq_len * binary_len:
        raise ValueError(f"Flattened sequence length mismatch: expected {seq_len * binary_len}, got {flattened_seq_len}")

    # Generate a base mask for the sequence
    mask = torch.ones(seq_len, seq_len, device=embeddings.device)

    # Expand for heads and batch size
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)

    # Reshape to match [batch_size * num_heads, seq_len, seq_len]
    mask = mask.reshape(batch_size * num_heads, seq_len, seq_len)

    logging.debug(f"Generated attention mask shape: {mask.shape}")
    return mask




# Top-K and Top-P Filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.tensor(filter_value, device=logits.device), logits)
    
    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    return logits


# Model Loading Function
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint and 'model_parameters' in checkpoint:
        # New model format with parameters included
        state_dict = checkpoint['state_dict']
        model_parameters = checkpoint['model_parameters']
    else:
        # Old model format without parameters
        state_dict = checkpoint
        model_parameters = None

    return state_dict, model_parameters


def text_to_float_sequence(input_text, max_seq_len=1024, device=device):
    byte_sequence = list(input_text.encode('utf-8')[:max_seq_len - 1])  # Reserve space for EOS
    byte_sequence.append(EOS_TOKEN)  # Append EOS token
    return torch.tensor(byte_sequence, dtype=torch.float32, device=device) / 256.0

def logits_to_tokens(logits, temperature=1.0, top_k=0, top_p=0.0, max_vocab=token_length):
    logging.debug(f"Logits to tokens shape: {logits.shape}, logits values: {logits}")

    logits = logits / temperature  # Apply temperature scaling

    # Clamp logits to ensure valid token values
    logits = logits.clamp(0, max_vocab - 1)

    # Apply top-k and top-p filtering
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Sample from the filtered logits
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1)
    logging.debug(f"next token id befoe clamp: {next_token_id}")

    return next_token_id.clamp(0, max_vocab-1)


def tokens_to_byte_sequence(tokens, max_vocab=token_length):
    """
    Converts tokens (floats or integers) to a sequence of byte-like values.
    """
    byte_tensor = tokens.view(-1)

    # Normalize float values to integers within the vocab range
    if byte_tensor.dtype in [torch.float32, torch.float64]:
        byte_tensor = (byte_tensor * max_vocab).long()

    # Stop at EOS token
    eos_mask = byte_tensor == EOS_TOKEN
    if eos_mask.any():
        eos_index = eos_mask.nonzero(as_tuple=True)[0][0].item()
        byte_tensor = byte_tensor[:eos_index]

    logging.debug(f"Generated byte sequence: {byte_tensor.tolist()}")
    return byte_tensor.tolist()




#GUI Implementation
class LanguageModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Binary Transformer Inference Program")

        # Initialize model as None
        self.model = None

        # Define Entry widgets for model path
        Label(root, text="Model Path:").pack(pady=(10, 0))
        self.model_path_entry = Entry(root, width=60)
        self.model_path_entry.pack(pady=(0, 10))

        Label(root, text="Architecture:").pack(pady=(0, 0))
        self.architecture_var = StringVar(value="matmul_free")
        architecture_menu = OptionMenu(root, self.architecture_var, "matmul_free", "mini_transformer")
        architecture_menu.pack(pady=(0, 10))

        # Select Folder Button
        self.select_button = Button(root, text="Select Model Folder", command=self.select_folder)
        self.select_button.pack(pady=(0, 10))

        # Model Parameters
        Label(root, text="Vocabulary Size:").pack(pady=(10, 0))
        self.vocab_size_entry = Entry(root, width=60)
        self.vocab_size_entry.pack(pady=(0, 10))
        self.vocab_size_entry.insert(0, "30000")  # Default value

        Label(root, text="Embedding Size:").pack(pady=(0, 0))
        self.embed_size_entry = Entry(root, width=60)
        self.embed_size_entry.pack(pady=(0, 10))
        self.embed_size_entry.insert(0, "60")  # Default value

        Label(root, text="Hidden Size:").pack(pady=(0, 0))
        self.hidden_size_entry = Entry(root, width=60)
        self.hidden_size_entry.pack(pady=(0, 10))
        self.hidden_size_entry.insert(0, "60")  # Default value

        Label(root, text="Nodes:").pack(pady=(0, 0))
        self.num_nodes_entry = Entry(root, width=60)
        self.num_nodes_entry.pack(pady=(0, 10))
        self.num_nodes_entry.insert(0, "4")  # Default value
        
        Label(root, text="Heads:").pack(pady=(0, 0))
        self.num_heads_entry = Entry(root, width=60)
        self.num_heads_entry.pack(pady=(0, 10))
        self.num_heads_entry.insert(0, "6")  # Default value

        # Input Text
        Label(root, text="Input Text:").pack(pady=(10, 0))
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack(pady=(0, 10))

        # Generation Parameters
        Label(root, text="Max Length:").pack(pady=(10, 0))
        self.max_length_entry = Entry(root, width=60)
        self.max_length_entry.pack(pady=(0, 10))
        self.max_length_entry.insert(0, "50")

        Label(root, text="Temperature:").pack(pady=(0, 0))
        self.temperature_entry = Entry(root, width=60)
        self.temperature_entry.pack(pady=(0, 10))
        self.temperature_entry.insert(0, "1.0")

        Label(root, text="Top-K:").pack(pady=(0, 0))
        self.top_k_entry = Entry(root, width=60)
        self.top_k_entry.pack(pady=(0, 10))
        self.top_k_entry.insert(0, "0")

        Label(root, text="Top-P:").pack(pady=(0, 0))
        self.top_p_entry = Entry(root, width=60)
        self.top_p_entry.pack(pady=(0, 10))
        self.top_p_entry.insert(0, "0.0")

        Label(root, text="Repetition Penalty:").pack(pady=(0, 0))
        self.repetition_penalty_entry = Entry(root, width=60)
        self.repetition_penalty_entry.pack(pady=(0, 10))
        self.repetition_penalty_entry.insert(0, "1.0")

        # Generate Button
        self.generate_button = Button(root, text="Generate Text", command=self.generate_text_callback)
        self.generate_button.pack(pady=(0, 10))

        # Output Box
        Label(root, text="Generated Output:").pack(pady=(10, 0))
        self.output_box = Text(root, height=10, width=60)
        self.output_box.pack(pady=(0, 10))
        
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {device}")

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Set model path
            model_path = os.path.join(folder_path, "binary_cascade_transformer.pth")

            # Update Entry widgets
            self.model_path_entry.delete(0, END)
            self.model_path_entry.insert(0, model_path)

            # Load model and "tokenizer"
            try:
                self.load_model_and_tokenizer(model_path)
                messagebox.showinfo("Success", "Model and byte-level binary tokenizer loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/tokenizer:\n{e}")

    def load_model_and_tokenizer(self, model_path):

        # Load model parameters from model_config.json
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Error", "model_config.json not found.")
            return

        model_parameters = load_model_parameters(config_path)

        #Update Entry widgets with loaded parameters
        self.vocab_size_entry.config(state='normal')
        self.vocab_size_entry.delete(0, END)
        self.vocab_size_entry.insert(0, str(model_parameters['vocab_size']))
        self.vocab_size_entry.config(state='readonly')

        self.embed_size_entry.config(state='normal')
        self.embed_size_entry.delete(0, END)
        self.embed_size_entry.insert(0, str(model_parameters['embed_size']))
        self.embed_size_entry.config(state='readonly')

        self.hidden_size_entry.config(state='normal')
        self.hidden_size_entry.delete(0, END)
        self.hidden_size_entry.insert(0, str(model_parameters['hidden_size']))
        self.hidden_size_entry.config(state='readonly')

        self.num_nodes_entry.config(state='normal')
        self.num_nodes_entry.delete(0, END)
        self.num_nodes_entry.insert(0, str(model_parameters['num_nodes']))
        self.num_nodes_entry.config(state='readonly')
        
        self.num_heads_entry.config(state='normal')
        self.num_heads_entry.delete(0, END)
        self.num_heads_entry.insert(0, str(model_parameters['num_heads']))
        self.num_heads_entry.config(state='readonly')
        
        if 'architecture' in model_parameters:
            architecture = model_parameters['architecture']
            self.architecture_var.set(model_parameters['architecture'])

            if architecture not in ['matmul_free', 'mini_transformer']:
                raise ValueError(f"Unsupported architecture: {architecture}")


        if architecture == 'matmul_free':
            model = BinaryCascadeTransformer(
                num_nodes=model_parameters['num_nodes'],
                hidden_size=model_parameters['hidden_size'],
                num_heads=model_parameters['num_heads'],
                max_seq_length=model_parameters['max_seq_length'],
                vocab_size=token_length,  # Byte embeddings cover values 0–255
                node_type='matmul_free',
                num_layers=model_parameters['num_layers']
            )
        elif architecture == 'mini_transformer':
            model = BinaryCascadeTransformer(
                num_nodes=model_parameters['num_nodes'],
                hidden_size=model_parameters['hidden_size'],
                num_heads=model_parameters['num_heads'],
                max_seq_length=model_parameters['max_seq_length'],
                vocab_size=token_length,
                node_type='mini_transformer',
                num_layers=model_parameters['num_layers']
            )
        else:
            raise ValueError(f"Unsupported architecture type: {architecture}")

        self.preprocessing_node = ModifiedTransformerNode(
            embed_size=model_parameters['embed_size'],
            hidden_size=model_parameters['hidden_size'],
            num_heads=model_parameters['num_heads'],
            num_layers=model_parameters['num_layers'],
            max_seq_length=model_parameters['max_seq_length']
        ).to(device)

        # Load state_dict
        state_dict, _ = load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Update class attributes
        self.model = model

    def generate_text_callback(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return

        input_text = self.input_box.get("1.0", END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some input text.")
            return

        # Retrieve generation parameters
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get())
            top_p = float(self.top_p_entry.get())
            repetition_penalty = float(self.repetition_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid generation parameters.")
            return

        # Start generation in a separate thread to keep GUI responsive
        threading.Thread(
            target=self.generate_and_display,
            args=(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
        ).start()

    def generate_and_display(self, input_text, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            output = self.generate_text_gui(
                model=self.model,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            logging.debug(f"Generated text output: {output}")
            self.output_box.delete("1.0", END)
            self.output_box.insert(END, output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text:\n{e}")

    # Text Generation Function
    def generate_text_gui(self, model, input_text, max_length=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
        """
        Generate text using the updated model with [vocab_size, 2] logits.
        """
        model.eval()
        model.to(device)
        logging.debug(f"Input text: {input_text}")
        self.device = device

        # Preprocess the input text
        input_ids = preprocess_text(input_text)  # Shape: [seq_len, binary_len]
        logging.debug(f"Input ids shape before adding batch dimension: {input_ids.shape}")

        # Add batch dimension
        input_ids = input_ids.unsqueeze(0)  # Shape: [1, seq_len, binary_len]
        logging.debug(f"Input ids shape after adding batch dimension: {input_ids.shape}")

        # Forward pass through the model
        with torch.no_grad():
            logits, _ = self.model(input_ids.to(self.device))  # Shape: [batch_size, seq_len, vocab_size, 2]
            logging.debug(f"Logits shape: {logits.shape}")

            # Select the most probable class
            predictions = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
            logging.debug(f"Predictions shape: {predictions.shape}")
            logging.debug(f"Logits: {logits}")
            logging.debug(f"Predictions (after argmax): {predictions}")


            # Flatten predictions for decoding
            predictions = predictions.view(-1)  # Shape: [total_positions]
            logging.debug(f"flattened Predictions: {predictions}")

            # Decode predictions back into text
            output_text = decode_binary_sequence(predictions.tolist())
            logging.debug(f"Decoded output text: {output_text}")

                # Convert binary samples back to text
            decoded_text = decode_binary_sequence(predictions.squeeze().tolist())
            logging.info(f"Decoded Text from Sampled Binary: {decoded_text}")

            messagebox.showinfo(
                    "binary Embedding Test",
                    f"binary embeddings and probabilities computed successfully.\n"
                    f"Decoded Text: {decoded_text}\n"
                    f"Check logs for tensor values."
                )

        return output_text



def main():
    root = Tk()
    gui = LanguageModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
