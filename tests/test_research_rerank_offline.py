"""The no-download setting reaches the actual model-loading boundary."""
import pytest

import rerank


class _NoEmbedder:
    def available(self):
        return False


@pytest.mark.parametrize("allow_download", [False, True])
def test_cross_encoder_honors_download_policy_for_cached_models(allow_download):
    def load(model, *, device, local_files_only=False):
        if local_files_only != (not allow_download):
            raise OSError("wrong model download policy")
        return object()

    scorer = rerank.Reranker("cross", embedder=_NoEmbedder(), cross_encoder=load,
                             allow_download=allow_download, is_cached=lambda model: True)
    assert scorer.backend == "cross"
