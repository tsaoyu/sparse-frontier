from .efficient_prefilling import (
    DenseAttention,
    VerticalAndSlashAttentionMInference,
    BlockSparseAttentionMInference,
    FlexPrefill,
)
from .efficient_decoding import QuestAttention
from .kv_compression import SnapKVCompression, AdaSnapKVCompression
from .rl_sparse_attention import RLSparseAttention
from sparse_frontier.utils import GlobalSettings
from .handler import AttentionHandler


ATTENTION_REGISTRY = {
    'dense': DenseAttention,
    'vertical_and_slash': VerticalAndSlashAttentionMInference,
    'block_sparse': BlockSparseAttentionMInference,
    'snapkv': SnapKVCompression,
    'ada_snapkv': AdaSnapKVCompression,
    'quest': QuestAttention,
    'flexprefill': FlexPrefill,
    'rl_sparse': RLSparseAttention,
}


def get_attention():
    return GlobalSettings.get('ATTENTION', DenseAttention())


def get_attention_handler() -> AttentionHandler:
    return GlobalSettings.get('ATTENTION_HANDLER')


def configure_attention():
    cfg = GlobalSettings.get('cfg')

    attention_args = cfg.attention.get('args', {})

    if cfg.attention.name == 'quest':
        block_size = attention_args.page_size
    else:
        block_size = cfg.kv_cache_block_size
    
    # Configure attention handler
    attention_handler = AttentionHandler(
        tp_size=cfg.tp,
        model_q_heads=cfg.model.num_q_heads,
        model_kv_heads=cfg.model.num_kv_heads,
        model_layers=cfg.model.num_layers,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=cfg.max_output_tokens,
        block_size=block_size,
    )
    GlobalSettings.set('ATTENTION_HANDLER', attention_handler)
    
    if cfg.attention.name == 'quest':
        extra_args = {
            'num_layers': cfg.model.num_layers,
            'max_input_tokens': cfg.max_input_tokens,
            'max_output_tokens': cfg.max_output_tokens,
        }
    else:
        extra_args = {}
    
    attention = ATTENTION_REGISTRY[cfg.attention.name](**attention_args, **extra_args)
    GlobalSettings.set('ATTENTION', attention)
