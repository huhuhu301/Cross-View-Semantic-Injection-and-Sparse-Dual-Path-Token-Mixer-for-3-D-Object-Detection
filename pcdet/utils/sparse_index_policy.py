# Modified by RV-SDTM contributors; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Select spconv's indice-table algorithm before its PyTorch operators load."""
import inspect
import sys


use_direct_table = False


def select(enabled):
    global use_direct_table
    ops = sys.modules.get('spconv.pytorch.ops')
    if ops is not None:
        # spconv 2.3 captures this setting as a function default at import time.
        params = inspect.signature(ops.get_indice_pairs_implicit_gemm).parameters
        if 'direct_table' in params and params['direct_table'].default != enabled:
            raise RuntimeError('The sparse indice policy is already bound. Build this model '
                               'in a fresh process before importing spconv.pytorch operators.')
    use_direct_table = enabled
