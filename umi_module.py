from asyncio import gather
from threading import Condition
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class AdaptivePooling1D(nn.Module):
    """1D Adaptive Average Pooling to align sequence lengths"""
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)
        x = x.transpose(1, 2)  # (batch_size, feature_dim, seq_len)
        x = self.adaptive_pool(x)  # (batch_size, feature_dim, target_length)
        return x.transpose(1, 2)


class ModalityProjection(nn.Module):
    """Project modality features to common dimension"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.projection(x))


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = query.size()
        key_seq_len = key.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask shape logic:
            # Expecting mask where 1 is keep, 0 is mask out.
            
            if mask.dim() == 2: # (batch_size, key_seq_len)
                mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, seq_len, key_seq_len)
            elif mask.dim() == 3:  # (batch_size, 1, key_seq_len) or similar
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, seq_len, key_seq_len)
            elif mask.dim() == 4:  # Already in correct shape
                pass
            
            # Apply mask: positions with 0 are masked out (-inf)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        return self.layer_norm(output + query)


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with residual connection"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)


class CrossModalImputationBlock(nn.Module):
    """Cross-Modal Imputation Block for generating missing modality features"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention
        # Pass the mask to attention
        attended = self.cross_attention(query, key_value, key_value, mask)
        # Feed-forward network
        output = self.ffn(attended)
        return output


class AdaptiveGating(nn.Module):
    """Adaptive gating mechanism for feature fusion"""
    def __init__(self, d_model: int, num_modalities: int):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(num_modalities + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mask: torch.Tensor, modality_embedding: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, num_modalities)
        # modality_embedding: (batch_size, d_model)
        gate_input = torch.cat([mask, modality_embedding], dim=-1)
        gate = self.gate_network(gate_input)
        return gate.squeeze(-1)  # (batch_size,)


class UMI(nn.Module):
    """
    Unified Missing-modality Imputation Module
    """
    
    def __init__(self, 
                 num_modalities: int = 4,
                 input_dims: List[int] = [128, 128, 128, 128], 
                 sequence_lengths: List[int] = [8, 16, 16, 16], 
                 d_model: int = 256,
                 target_length: int = 16,
                 num_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.target_length = target_length
        
        # Step 1: Feature Alignment and Projection
        self.adaptive_pooling = nn.ModuleList([
            AdaptivePooling1D(target_length) for _ in range(num_modalities)
        ])
        
        self.modality_projections = nn.ModuleList([
            ModalityProjection(input_dims[i], d_model) for i in range(num_modalities)
        ])
        
        # Step 2: Mask Encoding
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, target_length, d_model)
        )
        
        self.present_token = nn.Parameter(torch.randn(d_model))
        self.absent_token = nn.Parameter(torch.randn(d_model))
        
        # Step 3: Cross-Modal Imputation Blocks
        self.imputation_blocks = nn.ModuleList([
            CrossModalImputationBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_modalities)
        ])
        
        self.query_constructors = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(num_modalities)
        ])
        
        # Step 4: Adaptive Feature Fusion
        self.adaptive_gating = nn.ModuleList([
            AdaptiveGating(d_model, num_modalities) for _ in range(num_modalities)
        ])
        
        self._init_parameters()
    
    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                modality_features: List[torch.Tensor], 
                mask: torch.Tensor,
                key_padding_masks: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Args:
            modality_features: List of tensors.
            mask: Availability mask (batch, num_modalities). 1=Present, 0=Absent.
            key_padding_masks: List of tensors, one per modality.
                               Shape: (batch_size, source_seq_len).
                               Values: True/1 for Padding (Ignore), False/0 for Valid.
                               (Standard PyTorch convention)
        """
        batch_size = modality_features[0].size(0)
        device = modality_features[0].device
        
        # Handle default key_padding_masks if None (assume all valid)
        if key_padding_masks is None:
            key_padding_masks = []
            for feats in modality_features:
                # False means valid (not padding)
                key_padding_masks.append(torch.zeros(batch_size, feats.size(1), dtype=torch.bool, device=device))

        # Step 1: Feature Alignment and Projection
        aligned_features = []
        projected_features = []
        
        # We also need to process the padding masks because pooling changes the length
        processed_padding_masks = [] 

        for i, (features, pooling, projection) in enumerate(
            zip(modality_features, self.adaptive_pooling, self.modality_projections)
        ):
            # Align sequence length
            aligned = pooling(features) 
            aligned_features.append(aligned)
            
            # Project
            projected = projection(aligned)
            projected_features.append(projected)
            
            # Process Padding Mask: Resize to target_length
            # Input mask: (B, Source_Len), True=Pad
            # We invert it to (B, Source_Len), 1.0=Valid, 0.0=Pad for interpolation
            curr_mask = (~key_padding_masks[i]).float().unsqueeze(1) # (B, 1, Source_Len)
            
            # Interpolate mask to target length (Nearest neighbor to keep binary nature)
            resized_mask = F.interpolate(curr_mask, size=self.target_length, mode='nearest') # (B, 1, Target_Len)
            processed_padding_masks.append(resized_mask.squeeze(1)) # (B, Target_Len)
        
        # Step 2: Mask Encoding and Conditioning
        conditioning_tokens = []
        for i in range(self.num_modalities):
            present_mask = mask[:, i].unsqueeze(-1)
            absent_mask = 1 - present_mask
            present_cond = present_mask * self.present_token.unsqueeze(0)
            absent_cond = absent_mask * self.absent_token.unsqueeze(0)
            conditioning = present_cond + absent_cond
            conditioning_tokens.append(conditioning)
        
        # Step 3: Cross-Modal Imputation
        imputed_features = []
        
        for i in range(self.num_modalities):
            # Create query
            modality_embedding = self.modality_embeddings[i].unsqueeze(0).expand(batch_size, -1, -1)
            Condition_embedding = conditioning_tokens[i].unsqueeze(1).expand(-1, self.target_length, -1)
            query_input = torch.cat([Condition_embedding, modality_embedding], dim=-1)
            query = self.query_constructors[i](query_input)
            
            # Get available modalities for key-value
            available_features_list = []
            available_masks_list = []
            
            for j in range(self.num_modalities):
                if j != i:
                    # 1. Features
                    modality_mask = mask[:, j].unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
                    masked_features = projected_features[j] * modality_mask
                    available_features_list.append(masked_features)
                    
                    # 2. Attention Masks
                    # We need to combine:
                    #   a) Sequence Padding Mask (processed_padding_masks[j]): 1=Valid, 0=Pad
                    #   b) Modality Availability Mask (mask[:, j]): 1=Present, 0=Absent
                    
                    # seq_mask: (B, Target_Len), 1=Valid
                    seq_mask = processed_padding_masks[j] 
                    # mod_avail: (B, 1)
                    mod_avail = mask[:, j].unsqueeze(-1)
                    
                    # Combined: If modality is absent, whole sequence is 0. 
                    # If present, only padding is 0.
                    combined_mask = seq_mask * mod_avail 
                    available_masks_list.append(combined_mask)
            
            if available_features_list:
                # Concatenate features along sequence dimension
                key_value = torch.cat(available_features_list, dim=1) # (B, Total_Len, D)
                
                # Concatenate masks along sequence dimension
                # Shape: (B, Total_Len) where 1=Attend, 0=Ignore
                attention_mask = torch.cat(available_masks_list, dim=1)
                
                # Cross-modal imputation with Mask
                imputed = self.imputation_blocks[i](query, key_value, mask=attention_mask)
                
                imputed_features.append(imputed)
            else:
                imputed_features.append(torch.zeros_like(projected_features[i]))
        
        # Step 4: Adaptive Feature Fusion
        final_features = []
        
        for i in range(self.num_modalities):
            present_mask = mask[:, i].unsqueeze(-1).unsqueeze(-1)
            absent_mask = 1 - present_mask
            
            original = projected_features[i] * present_mask
            imputed = imputed_features[i]
            
            if present_mask.sum() > 0:
                modality_embedding = self.modality_embeddings[i].unsqueeze(0).expand(batch_size, -1, -1).mean(dim=1)
                gate = self.adaptive_gating[i](mask, modality_embedding)
                gate = gate.unsqueeze(-1).unsqueeze(-1)
                
                fused_present = (1 - gate) * original + gate * imputed
                fused_absent = imputed
                final = present_mask * fused_present + absent_mask * fused_absent
            else:
                final = imputed
            
            final_features.append(final)
        
        return final_features

