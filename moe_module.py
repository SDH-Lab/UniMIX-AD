import torch
import torch.nn.functional as F
import torch.nn as nn
from fmoe.transformer import _Expert
from fmoe.layers import FMoE, _fmoe_general_global_forward, mark_module_parallel_comm
from fmoe.functions import ensure_comm, Slice, AllGather
from fmoe.gates import NaiveGate

import tree

from fmoe.gates import NoisyGate

class FixedFMoE(nn.Module):
    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None, slice_group=None, moe_group=None, top_k=2, gate=NaiveGate, expert=None, gate_hook=None, mask=None, mask_dict=None):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        if mp_group is not None:

            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp, expert_indices=None):
        moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))

        assert all([batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp, expert_indices)

        # Reshape gate_top_k_idx to be 2-dimensional
        gate_top_k_idx = gate_top_k_idx.view(moe_inp.shape[0], self.top_k)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        self.gate.set_topk_indicates(gate_top_k_idx)

        if self.mask is not None and self.mask_dict is not None:
            def delete_mask_func(tensor):
                tensor = tensor[self.mask == 0, :]
                return tensor
            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size, experts=self.experts)

        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(mask.shape[0], self.top_k, dim, device=tensor.device, dtype=tensor.dtype)
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)
        # Compute outputs from all experts only during training (for MCG loss)
        all_expert_outputs = None
        if self.training:
            all_expert_outputs = self._compute_all_expert_outputs(moe_inp, gate_top_k_idx, moe_outp)

        gate_score = gate_score.view(-1, 1, self.top_k)
        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all([batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]), "MoE outputs must have the same batch size"
        return moe_outp, all_expert_outputs

    def _compute_all_expert_outputs(self, moe_inp, gate_top_k_idx, moe_outp_topk):
        """
        Compute outputs from all experts (both selected and non-selected).
        Combines them into a feature embedding with shape [batch_size, num_expert, dim].
        
        Args:
            moe_inp: Input to the MoE layer [B, d_model]
            gate_top_k_idx: Indices of top-K experts [B, top_k]
            moe_outp_topk: Outputs from top-K experts [B, top_k, dim]
        
        Returns:
            all_expert_outputs: Tensor of shape [B, num_expert, dim]
        """
        batch_size = moe_inp.shape[0]
        dim = moe_outp_topk.shape[-1]
        
        # Initialize output tensor for all experts
        all_expert_outputs = torch.zeros(batch_size, self.num_expert, dim, 
                                         device=moe_inp.device, dtype=moe_inp.dtype)
        
        # Create mask for non-selected experts
        # Start with all experts being non-selected
        non_selected_mask = torch.ones(batch_size, self.num_expert, device=moe_inp.device, dtype=torch.bool)
        # Mark selected experts as False
        non_selected_mask.scatter_(1, gate_top_k_idx, False)
        
        # Fill in the top-K expert outputs
        all_expert_outputs.scatter_(1, gate_top_k_idx.unsqueeze(-1).expand(-1, -1, dim), moe_outp_topk)
        
        # Compute outputs from all non-selected experts
        if self.experts_fused:
            # Handle fused experts (_Expert object)
            self._compute_fused_expert_outputs(moe_inp, non_selected_mask, all_expert_outputs)
        else:
            # Handle non-fused experts (ModuleList)
            for expert_idx in range(self.num_expert):
                # Get samples where this expert is non-selected
                expert_non_selected = non_selected_mask[:, expert_idx]
                
                if expert_non_selected.sum() == 0:
                    continue
                
                # Get input for non-selected samples
                expert_input = moe_inp[expert_non_selected]
                
                # Compute expert output
                expert_output = self.experts[expert_idx](expert_input)
                
                # Fill in the outputs for non-selected samples
                all_expert_outputs[expert_non_selected, expert_idx, :] = expert_output
        
        return all_expert_outputs
    
    def _compute_fused_expert_outputs(self, moe_inp, non_selected_mask, all_expert_outputs):
        """
        Compute outputs for non-selected experts when using fused experts (_Expert class).
        
        Args:
            moe_inp: Input to the MoE layer [B, d_model]
            non_selected_mask: Boolean mask [B, num_expert] indicating non-selected experts
            all_expert_outputs: Output tensor to fill [B, num_expert, dim]
        """
        # Access the fused expert's linear layers
        htoh4_weight = self.experts.htoh4.weight  # [num_expert, d_hidden, d_model]
        htoh4_bias = self.experts.htoh4.bias if self.experts.htoh4.bias is not None else None  # [num_expert, d_hidden]
        h4toh_weight = self.experts.h4toh.weight  # [num_expert, d_model, d_hidden]
        h4toh_bias = self.experts.h4toh.bias if self.experts.h4toh.bias is not None else None  # [num_expert, d_model]
        
        # Compute outputs from all non-selected experts
        for expert_idx in range(self.num_expert):
            # Get samples where this expert is non-selected
            expert_non_selected = non_selected_mask[:, expert_idx]
            
            if expert_non_selected.sum() == 0:
                continue
            
            # Get input for non-selected samples
            expert_input = moe_inp[expert_non_selected]  # [N, d_model]
            
            # Compute expert output manually using the fused expert's parameters
            # First layer: htoh4 (input -> hidden)
            x = torch.matmul(expert_input, htoh4_weight[expert_idx].t())  # [N, d_hidden]
            if htoh4_bias is not None:
                x = x + htoh4_bias[expert_idx]
            x = self.experts.activation(x)
            expert_output = torch.matmul(x, h4toh_weight[expert_idx].t())
            if h4toh_bias is not None:
                expert_output = expert_output + h4toh_bias[expert_idx]
            
            # Fill in the outputs for non-selected samples
            all_expert_outputs[expert_non_selected, expert_idx, :] = expert_output

class FMoETransformerMLP(FixedFMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        n_router = 1,
        gate='AddtionalNoisyGate', # NaiveGate
        world_size=1,
        top_k=2,
        **kwargs
    ):
        if type(gate) is str:
            gate = eval(gate)
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.n_router = n_router
        self.all_gates = nn.ModuleDict({f'{i}': gate(d_model, num_expert, world_size, top_k) for i in range(n_router)})
        self.gate = self.all_gates[f'{0}']

        self.mark_parallel_comm(expert_dp_comm)
        
        # Build modality combination graph for MCG contrastive learning
        self.mcg_adjacency_matrix = self._build_modality_combination_graph()
        self.temperature = 0.07  # Temperature parameter for InfoNCE loss

    def forward(self, inp: torch.Tensor, expert_indices=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        
        output, all_expert_outputs = super().forward(inp, expert_indices=expert_indices)
        
        # Calculate MCG contrastive loss only during training
        MCG_loss = None
        if self.training and all_expert_outputs is not None:
            batch_size = original_shape[0]
            seq_len = original_shape[1]
            num_expert = all_expert_outputs.shape[1]
            dim = all_expert_outputs.shape[2]
            all_expert_outputs = all_expert_outputs.reshape(batch_size, seq_len, num_expert, dim)
            MCG_loss = self.calculate_MCG_contrastive_loss(all_expert_outputs)


        return output.reshape(original_shape), MCG_loss
    
    def _build_modality_combination_graph(self):
        """
        Build adjacency matrix for modality combination graph.
        For 4 modalities (T, M, F, P), we have 15 combinations (2^4 - 1).
        Two combinations are connected if they differ by exactly one modality.
        
        Returns:
            adjacency_matrix: torch.Tensor of shape [15, 15] where 1 indicates edge, 0 no edge
        """
        from itertools import combinations
        
        # Define all 15 modality combinations (excluding empty set)
        modalities = ['T', 'M', 'F', 'P']
        all_combinations = []
        
        # Generate combinations in order: 4 modalities, then 3, then 2, then 1
        for i in range(len(modalities), 0, -1):
            comb = list(combinations(modalities, i))
            all_combinations.extend(comb)
        
        # Convert to sets for easier comparison
        combination_sets = [set(comb) for comb in all_combinations]
        
        # Build adjacency matrix
        num_combinations = len(combination_sets)  # Should be 15
        adjacency_matrix = torch.zeros(num_combinations, num_combinations)
        
        # Two combinations are connected if they differ by exactly one modality
        for i in range(num_combinations):
            for j in range(i + 1, num_combinations):
                set_i = combination_sets[i]
                set_j = combination_sets[j]
                
                # Check if they differ by exactly one modality
                symmetric_diff = set_i.symmetric_difference(set_j)
                if len(symmetric_diff) == 1:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
        
        return adjacency_matrix
    
    def print_mcg_graph_structure(self):
        """
        Print the structure of the modality combination graph for debugging.
        Shows all 15 combinations and their connections.
        """
        from itertools import combinations as iter_combinations
        
        modalities = ['T', 'M', 'F', 'P']
        all_combinations = []
        for i in range(len(modalities), 0, -1):
            comb = list(iter_combinations(modalities, i))
            all_combinations.extend(comb)
        
        print("\n=== Modality Combination Graph Structure ===")
        print(f"Total nodes: {len(all_combinations)}")
        print("\nNode Index -> Modality Combination -> Adjacent Nodes:")
        
        for idx, comb in enumerate(all_combinations):
            comb_str = ''.join(sorted(comb))
            adjacent_indices = torch.where(self.mcg_adjacency_matrix[idx] > 0)[0].tolist()
            adjacent_combs = [(''.join(sorted(all_combinations[i]))) for i in adjacent_indices]
            print(f"  {idx:2d}: {comb_str:4s} -> {adjacent_combs}")
        print("=" * 50 + "\n")
    
    def calculate_MCG_contrastive_loss(self, all_expert_outputs):
        """
        Fully vectorized InfoNCE per token without Python loops.
        Uses masked log-softmax for maximum efficiency.
        
        This implementation computes InfoNCE loss for all tokens simultaneously:
        S: [N, 15, 15] similarity matrix where N=B×L
        A: [15, 15] adjacency matrix for positive pairs  
        
        For each anchor-positive pair (i,j), the loss is:
        -S[i,j] + logsumexp(S[i,k] for all k≠i)
        
        Args:
            all_expert_outputs: [B, L, E, D] - batch, seq_len, num_experts, dim
            
        Returns:
            loss: scalar tensor - average InfoNCE loss over all positive pairs
        """
        B, L, E, D = all_expert_outputs.shape
        device = all_expert_outputs.device
        num_specialized_experts = 15  # Exclude buffer expert
        
        # Reshape to process all tokens together [B*L, E, D]
        num_tokens = B * L
        specialized_outputs = all_expert_outputs[:, :, :num_specialized_experts, :].reshape(num_tokens, num_specialized_experts, D)
        specialized_outputs = F.normalize(specialized_outputs, p=2, dim=2)
        
        # Compute similarity matrix with temperature scaling [N, 15, 15]
        S = torch.bmm(specialized_outputs, specialized_outputs.transpose(1, 2)) / self.temperature
        
        # Get adjacency matrix and precompute positive pairs
        A = self.mcg_adjacency_matrix.to(device).bool()  # [15, 15]
        anchor_indices, pos_indices = torch.where(A)  # Get all positive pairs
        
        # Early exit if no positive pairs (return 1D tensor for DataParallel compatibility)
        if len(anchor_indices) == 0:
            return torch.zeros(1, device=device, requires_grad=True)
        
        # Create identity mask for self-connections
        eye_mask = torch.eye(num_specialized_experts, device=device, dtype=torch.bool)
        not_self_mask = ~eye_mask  # [15, 15]
        
        # For numerical stability: subtract row-wise max from valid entries only
        S_for_max = S.masked_fill(eye_mask.unsqueeze(0), float('-inf'))  # Mask diagonal
        row_max, _ = S_for_max.max(dim=-1, keepdim=True)  # [N, 15, 1]
        row_max = row_max.masked_fill(row_max == float('-inf'), 0.0)  # Handle empty rows
        S_stable = S - row_max  # [N, 15, 15]
        
        # Compute denominator using logsumexp over non-self entries
        exp_S_stable = torch.exp(S_stable)  # [N, 15, 15]
        exp_S_masked = exp_S_stable.masked_fill(eye_mask.unsqueeze(0), 0.0)  # Zero diagonal
        denom_sum = exp_S_masked.sum(dim=-1, keepdim=False)  # [N, 15]
        log_denom = torch.log(denom_sum.clamp(min=1e-8)) + row_max.squeeze(-1)  # [N, 15]
        
        # Extract positive similarities and corresponding denominators
        pos_sims = S[:, anchor_indices, pos_indices]  # [N, num_pos_pairs]  
        pos_log_denoms = log_denom[:, anchor_indices]  # [N, num_pos_pairs]
        
        # InfoNCE loss: -positive_sim + log_denominator
        infonce_losses = -pos_sims + pos_log_denoms  # [N, num_pos_pairs]
        
        # Average over all positive pairs across all tokens
        total_loss = infonce_losses.mean()  # More numerically stable than sum/count
        
        # For DataParallel compatibility, return tensor with at least 1 dimension
        # This prevents the "gather scalars" warning in multi-GPU setups
        if total_loss.dim() == 0:
            total_loss = total_loss.unsqueeze(0)
        
        return total_loss

   
    def set_full_modality(self, is_full_modality):
        for gate in self.all_gates.values():
            if hasattr(gate, 'set_full_modality'):
                gate.set_full_modality(is_full_modality)


class AddtionalNoisyGate(NoisyGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)
        self.topk_logits = []
        self.indicates = None
        self.is_full_modality = False

    def set_topk_logit(self, logit):
        self.topk_logits.append(logit)
    
    def get_topk_logit(self, clear = True):
        topk_logit = self.topk_logits
        if clear:
            self.topk_logits = None
        return topk_logit

    def set_topk_indicates(self, indicate):
        self.indicates = indicate
        
    def get_topk_indicate(self, clear = True):
        topk_indicate = self.indicates
        if clear:
            self.indicates = None
        return topk_indicate
    
    def set_loss(self, loss):
        # Ensure loss is always a 1D tensor for DataParallel compatibility
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
    
    def set_full_modality(self, is_full_modality):
        self.is_full_modality = is_full_modality

    def forward(self, inp, expert_indices=None):
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
        loss = 0

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        if (expert_indices != None) & (expert_indices.sum() > 0):
            batch_size = inp.shape[0]
            num_experts = expert_indices.shape[0]
            
            repeats = batch_size // num_experts
            remainder = batch_size % num_experts

            if repeats > 0:
                expert_indices_expanded = expert_indices.repeat(repeats, 1).T.reshape(-1)

            else:
                expert_indices_expanded = torch.tensor([], dtype=expert_indices.dtype, device=expert_indices.device)

            if remainder > 0:
                expert_indices_expanded = torch.cat([expert_indices_expanded, torch.tensor([expert_indices[-1]]*remainder).to(expert_indices.device)])
            
            full_modality_mask_expanded = expert_indices_expanded == 0

        if expert_indices.sum() > 0:
            expert_idx_loss = F.cross_entropy(logits[~full_modality_mask_expanded], expert_indices_expanded[~full_modality_mask_expanded])
            loss += expert_idx_loss
        
        self.set_topk_logit(top_k_indices)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            )
        else:
            load = self._gates_to_load(gates)
        
        if (expert_indices != None):
            full_modality_mask = expert_indices == 0
            if full_modality_mask.sum() == len(full_modality_mask):
                load = load.sum(0) if self.training else load
                importance = gates.sum(0) if self.training else gates.sum(0)
                loss += self.cv_squared(importance) + self.cv_squared(load)
            else:
                importance_1 = gates[full_modality_mask_expanded, :].sum(0) if self.training else gates.sum(0)
                load_1 = load[full_modality_mask_expanded, :].sum(0) if self.training else load
                loss_1 = self.cv_squared(importance_1) + self.cv_squared(load_1)

                importance_2 = gates[~full_modality_mask_expanded, 1:].sum(0) if self.training else gates.sum(0)
                load_2 = load[~full_modality_mask_expanded, 1:].sum(0) if self.training else load
                loss_2 = self.cv_squared(importance_2) + self.cv_squared(load_2)

                loss = loss + loss_1 + loss_2
        else:
            load = load.sum(0) if self.training else load
            importance = gates.sum(0) if self.training else gates.sum(0)
            loss += self.cv_squared(importance) + self.cv_squared(load)
        
        self.set_loss(loss)
        
        return (
            top_k_indices.contiguous().view(-1),
            top_k_gates.contiguous().unsqueeze(1),
        )