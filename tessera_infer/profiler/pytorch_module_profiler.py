#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ModuleHookProfiler:
    """
    PyTorch模块级别的profiling工具
    使用hooks来监控每个模块的forward执行时间
    """
    
    def __init__(self, model: nn.Module, timer_manager=None):
        self.model = model
        self.timer_manager = timer_manager
        self.hooks = []
        self.module_times = defaultdict(list)
        self.module_names = {}
        self.logger = logging.getLogger(__name__)
        
    def register_hooks(self, prefix="model"):
        """为模型的所有子模块注册hooks"""
        self.hooks.clear()
        self.module_names.clear()
        self.module_times.clear()
        
        # 为每个命名模块注册hook
        for name, module in self.model.named_modules():
            if name == "":  # Skip root module
                continue
                
            full_name = f"{prefix}.{name}" if name else prefix
            self.module_names[id(module)] = full_name
            
            # 注册前向hook
            hook = module.register_forward_hook(self._forward_hook)
            self.hooks.append(hook)
            
        self.logger.debug(f"Registered hooks for {len(self.hooks)} modules")
    
    def _forward_hook(self, module, input, output):
        """Forward hook function"""
        module_id = id(module)
        module_name = self.module_names.get(module_id, f"unknown_{module_id}")
        
        # 记录时间 - 这里我们在hook里无法直接计时，需要在外部控制
        # 所以这个hook主要用于标记模块执行
        pass
    
    def profile_forward_pass(self, *inputs, detailed=True):
        """
        Profiling模型的一次forward pass
        """
        results = {}
        
        if detailed and self.timer_manager:
            # 详细profiling每个主要模块
            with self.timer_manager.get_timer("main").timer("model_forward_total"):
                
                # Profile s2_backbone
                if hasattr(self.model, 's2_backbone'):
                    with self.timer_manager.get_timer("main").timer("s2_backbone"):
                        s2_repr = self.model.s2_backbone(inputs[0])
                    results['s2_repr'] = s2_repr
                
                # Profile s1_backbone 
                if hasattr(self.model, 's1_backbone'):
                    with self.timer_manager.get_timer("main").timer("s1_backbone"):
                        s1_repr = self.model.s1_backbone(inputs[1])
                    results['s1_repr'] = s1_repr
                
                # Profile fusion
                if hasattr(self.model, 'fusion_method'):
                    with self.timer_manager.get_timer("main").timer("fusion_operation"):
                        if self.model.fusion_method == "sum":
                            fused = s2_repr + s1_repr
                        elif self.model.fusion_method == "concat":
                            fused = torch.cat([s2_repr, s1_repr], dim=-1)
                        else:
                            raise ValueError(f"Unknown fusion method: {self.model.fusion_method}")
                    results['fused'] = fused
                
                # Profile dimension reduction
                if hasattr(self.model, 'dim_reducer'):
                    with self.timer_manager.get_timer("main").timer("dim_reducer"):
                        output = self.model.dim_reducer(fused)
                    results['final_output'] = output
                
                return output
        else:
            # Standard forward pass
            return self.model(*inputs)
    
    def profile_transformer_encoder(self, transformer_module, input_tensor, module_name="transformer"):
        """
        详细profiling TransformerEncoder的各个组件
        """
        if not self.timer_manager:
            return transformer_module(input_tensor)
            
        with self.timer_manager.get_timer("main").timer(f"{module_name}_total"):
            x = input_tensor
            
            # Check module data type and ensure input tensor matches
            module_dtype = next(transformer_module.parameters()).dtype
            if x.dtype != module_dtype:
                x = x.to(module_dtype)
            
            # Profile embedding if exists
            if hasattr(transformer_module, 'embedding'):
                with self.timer_manager.get_timer("main").timer(f"{module_name}_embedding"):
                    bands = x[:, :, :-1]  # All columns except last one
                    doy = x[:, :, -1]     # Last column is DOY
                    bands_embedded = transformer_module.embedding(bands)
            
            # Profile temporal encoding if exists
            if hasattr(transformer_module, 'temporal_encoder'):
                with self.timer_manager.get_timer("main").timer(f"{module_name}_temporal_encoding"):
                    temporal_encoding = transformer_module.temporal_encoder(doy)
                    # Ensure both tensors have the same dtype before addition
                    if bands_embedded.dtype != temporal_encoding.dtype:
                        temporal_encoding = temporal_encoding.to(bands_embedded.dtype)
                    x = bands_embedded + temporal_encoding
            
            # Profile transformer encoder layers
            if hasattr(transformer_module, 'transformer_encoder'):
                with self.timer_manager.get_timer("main").timer(f"{module_name}_transformer_layers"):
                    # Ensure x matches the expected dtype for transformer encoder
                    if x.dtype != module_dtype:
                        x = x.to(module_dtype)
                    x = transformer_module.transformer_encoder(x)
            
            # Profile attention pooling if exists
            if hasattr(transformer_module, 'attn_pool'):
                with self.timer_manager.get_timer("main").timer(f"{module_name}_attention_pooling"):
                    # Ensure x matches the expected dtype for attention pooling
                    if x.dtype != module_dtype:
                        x = x.to(module_dtype)
                    x = transformer_module.attn_pool(x)
            
            return x
    
    def cleanup_hooks(self):
        """清理所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_names.clear()
        self.module_times.clear()


class DetailedInferenceProfiler:
    """
    详细的推理profiling工具，结合batch内部操作和模型内部模块
    """
    
    def __init__(self, model, timer_manager):
        self.model = model
        self.timer_manager = timer_manager
        self.module_profiler = ModuleHookProfiler(model, timer_manager)
        self.logger = logging.getLogger(__name__)
    
    def profile_batch_processing(self, batch_data, dataset, config, device, batch_idx):
        """
        详细profiling单个batch的处理过程
        """
        B = batch_data["s2_bands"].shape[0]
        
        # 1. Profile batch data extraction
        with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_data_extraction"):
            global_idxs = batch_data["global_idx"]
            s2_bands_batch = batch_data["s2_bands"].numpy()
            s2_masks_batch = batch_data["s2_masks"].numpy()
            s2_doys_batch = batch_data["s2_doys"].numpy()
            s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()
            s1_asc_doys_batch = batch_data["s1_asc_doys"].numpy()
            s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
            s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()
        
        results_list = []
        
        for r in range(config["repeat_times"]):
            repeat_timer_name = f"batch_{batch_idx}_repeat_{r}"
            
            with self.timer_manager.get_timer("main").timer(repeat_timer_name):
                
                # 2. Profile S2 sampling
                with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_s2_sampling"):
                    s2_input_np = self._sample_s2_batch(
                        s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        dataset.s2_band_mean, dataset.s2_band_std, 
                        config["sample_size_s2"], standardize=True
                    )
                
                # 3. Profile S1 sampling  
                with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_s1_sampling"):
                    s1_input_np = self._sample_s1_batch(
                        s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        dataset.s1_band_mean, dataset.s1_band_std,
                        config["sample_size_s1"], standardize=True
                    )
                
                # 4. Profile tensor conversion
                with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_tensor_conversion"):
                    s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                    s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                
                # 5. Profile detailed model forward pass
                with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_model_forward"):
                    z = self._profile_model_forward(s2_input, s1_input, batch_idx, r)
                
                # 6. Profile GPU synchronization if needed
                if device.type == 'cuda':
                    with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_gpu_sync"):
                        torch.cuda.synchronize(device)
                
                results_list.append(z)
        
        # 7. Profile result aggregation
        with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_result_aggregation"):
            if len(results_list) == 1:
                avg_repr = results_list[0]
            else:
                sum_repr = results_list[0]
                for z in results_list[1:]:
                    sum_repr += z
                avg_repr = sum_repr / float(len(results_list))
        
        # 8. Profile CPU transfer
        with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_cpu_transfer"):
            # Handle BF16 to FP32 conversion for numpy compatibility
            if avg_repr.dtype == torch.bfloat16:
                avg_repr_np = avg_repr.to(torch.float32).cpu().numpy()
            else:
                avg_repr_np = avg_repr.cpu().numpy()
            global_idxs_list = global_idxs.tolist()
        
        # 9. Profile result collection
        with self.timer_manager.get_timer("main").timer(f"batch_{batch_idx}_result_collection"):
            local_results = []
            for b in range(B):
                gidx = global_idxs_list[b]
                local_results.append((gidx, avg_repr_np[b]))
        
        return local_results
    
    def _profile_model_forward(self, s2_input, s1_input, batch_idx, repeat_idx):
        """
        详细profiling模型的forward pass
        """
        prefix = f"batch_{batch_idx}_repeat_{repeat_idx}"
        
        # Check model data type and convert inputs if necessary
        model_dtype = next(self.model.parameters()).dtype
        if model_dtype == torch.bfloat16 and s2_input.dtype != torch.bfloat16:
            s2_input = s2_input.to(torch.bfloat16)
            s1_input = s1_input.to(torch.bfloat16)
        elif model_dtype != torch.bfloat16 and s2_input.dtype == torch.bfloat16:
            s2_input = s2_input.to(torch.float32)
            s1_input = s1_input.to(torch.float32)
        
        # Profile S2 backbone (TransformerEncoder)
        with self.timer_manager.get_timer("main").timer(f"{prefix}_s2_backbone_total"):
            if hasattr(self.model.s2_backbone, 'embedding'):
                # 这是TransformerEncoder，进行详细profiling
                s2_repr = self.module_profiler.profile_transformer_encoder(
                    self.model.s2_backbone, s2_input, f"{prefix}_s2_transformer"
                )
            else:
                s2_repr = self.model.s2_backbone(s2_input)
        
        # Profile S1 backbone (TransformerEncoder)  
        with self.timer_manager.get_timer("main").timer(f"{prefix}_s1_backbone_total"):
            if hasattr(self.model.s1_backbone, 'embedding'):
                # 这是TransformerEncoder，进行详细profiling
                s1_repr = self.module_profiler.profile_transformer_encoder(
                    self.model.s1_backbone, s1_input, f"{prefix}_s1_transformer"
                )
            else:
                s1_repr = self.model.s1_backbone(s1_input)
        
        # Profile fusion operation
        with self.timer_manager.get_timer("main").timer(f"{prefix}_fusion"):
            if self.model.fusion_method == "sum":
                fused = s2_repr + s1_repr
            elif self.model.fusion_method == "concat":
                fused = torch.cat([s2_repr, s1_repr], dim=-1)
            else:
                raise ValueError(f"Unknown fusion method: {self.model.fusion_method}")
        
        # Profile dimension reduction
        with self.timer_manager.get_timer("main").timer(f"{prefix}_dim_reducer"):
            output = self.model.dim_reducer(fused)
        
        return output
    
    def _sample_s2_batch(self, s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean, band_std, sample_size_s2, standardize=True):
        """S2 sampling with detailed profiling"""
        import numpy as np
        
        B = s2_bands_batch.shape[0]
        out_list = []
        for b in range(B):
            valid_idx = np.nonzero(s2_masks_batch[b])[0]
            
            if len(valid_idx) == 0:
                valid_idx = np.arange(s2_bands_batch.shape[1])
            
            if len(valid_idx) < sample_size_s2:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
            idx_chosen = np.sort(idx_chosen)

            sub_bands = s2_bands_batch[b, idx_chosen, :]
            sub_doys = s2_doys_batch[b, idx_chosen]
            if standardize:
                sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

            out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])
            out_list.append(out_arr.astype(np.float32))

        return np.stack(out_list, axis=0).astype(np.float32)
    
    def _sample_s1_batch(self, s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean, band_std, sample_size_s1, standardize=True):
        """S1 sampling with detailed profiling"""
        import numpy as np
        
        B = s1_asc_bands_batch.shape[0]
        out_list = []
        for b in range(B):
            s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)
            s1_doys_all = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

            valid_mask = np.any(s1_bands_all != 0, axis=-1)
            valid_idx = np.nonzero(valid_mask)[0]
            if len(valid_idx) == 0:
                valid_idx = np.arange(s1_bands_all.shape[0])
            if len(valid_idx) < sample_size_s1:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
            idx_chosen = np.sort(idx_chosen)

            sub_bands = s1_bands_all[idx_chosen, :]
            sub_doys = s1_doys_all[idx_chosen]

            if standardize:
                sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

            out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])
            out_list.append(out_arr.astype(np.float32))

        return np.stack(out_list, axis=0).astype(np.float32)
    
    def cleanup(self):
        """清理资源"""
        self.module_profiler.cleanup_hooks()