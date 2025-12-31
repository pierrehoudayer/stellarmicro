try:
    import autograd.numpy as np
    from autograd import elementwise_grad as egrad
    AUTOGRAD_AVAILABLE = True
except Exception:
    import numpy as np  # type: ignore
    egrad = None
    AUTOGRAD_AVAILABLE = False

__all__ = ["np", "egrad", "AUTOGRAD_AVAILABLE"]
