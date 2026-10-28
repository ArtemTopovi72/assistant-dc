"""Embedding launch arguments, without starting the operator's server."""
from types import SimpleNamespace

import llama_backend
import pytest


def settings(**overrides):
    return SimpleNamespace(**dict(EMBED_BASE='http://127.0.0.1:8096',
        EMBED_CPU_SERVER='cpu/llama-server.exe', EMBED_GGUF='bge-m3.gguf',
        EMBED_MODEL='text-embedding-bge-m3', **overrides))


def test_legacy_cpu_config_keeps_the_existing_command_without_new_flags():
    assert llama_backend._embedder_command(settings()) == [
        'cpu/llama-server.exe', '-m', 'bge-m3.gguf', '--embedding', '--pooling', 'cls',
        '-c', '8192', '-b', '8192', '-ub', '8192', '--alias', 'text-embedding-bge-m3',
        '--host', '127.0.0.1', '--port', '8096', '-t', '8']


def test_gpu_config_keeps_endpoint_model_and_alias_and_disables_automatic_fitting():
    cfg=settings(EMBED_GPU_LAYERS='all',EMBED_FLASH_ATTN='on')
    cfg.EMBED_CPU_SERVER='cuda/llama-server.exe'
    command=llama_backend._embedder_command(cfg)
    assert command[0]=='cuda/llama-server.exe'
    assert command[1:]==llama_backend._embedder_command(settings())[1:]+[
        '--gpu-layers','all','--fit','off','--flash-attn','on','--parallel','1']


@pytest.mark.parametrize('layers', ['0', ' 0 ', ''])
def test_cpu_zero_does_not_send_cuda_flags_to_older_server_and_custom_port_is_preserved(layers):
    cfg=settings(EMBED_GPU_LAYERS=layers,EMBED_FLASH_ATTN='on')
    cfg.EMBED_BASE='http://localhost:8099'
    command=llama_backend._embedder_command(cfg)
    assert command[command.index('--port')+1]=='8099'
    assert '--gpu-layers' not in command and '--fit' not in command and '--flash-attn' not in command


def test_gpu_mode_without_flash_override_uses_auto():
    command=llama_backend._embedder_command(settings(EMBED_GPU_LAYERS='all'))
    assert command[command.index('--flash-attn')+1]=='auto'
