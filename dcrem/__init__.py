"""D-CREM: Deep Classifier-induced Reciprocal point for Multi-label OSR.

A learnable-feature extension of CREM.  Replaces the fixed RBF kernel with
an encoder g_θ(·) + ℓ₂ normalisation so that condition(★) (‖f(x)‖ = const)
holds, preserving the reciprocal-point induction mechanism while allowing
end-to-end training.
"""
