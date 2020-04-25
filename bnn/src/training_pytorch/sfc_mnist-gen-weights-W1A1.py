# Modified script to synthesize a more folded neural network.
# In the FINNTHESIZER, Number of layers is inferred from the length of the "SIMDCounts".

# Modified by Franco Caspe based on "mnist-gen-weights-W1A1.py" to pack SFC model's weights and thresholds.
# francocaspe@hotmail.com.

import os
import sys
import finnthesizer as fth

if __name__ == "__main__":
    bnnRoot = "."
    npzFile = bnnRoot + "/models/sfc_mnist-w1a1.npz"
    targetDirBin = bnnRoot + "/sfcW1A1"
    targetDirHLS = bnnRoot + "/sfcW1A1/hw"

    simdCounts = [32, 8, 16,  4]
    peCounts   = [16, 16, 8, 8]

    WeightsPrecisions_integer       = [1, 1, 1, 1]
    WeightsPrecisions_fractional    = [0, 0, 0, 0]
    
    InputPrecisions_integer         = [1, 1, 1, 1]
    InputPrecisions_fractional      = [0, 0, 0, 0]
    
    ActivationPrecisions_integer    = [1, 1, 1, 1]
    ActivationPrecisions_fractional = [0, 0, 0, 0]

    classes = map(lambda x: str(x), range(10))

    fth.convertFCNetwork(npzFile, targetDirBin, targetDirHLS, simdCounts, peCounts, WeightsPrecisions_fractional, ActivationPrecisions_fractional, InputPrecisions_fractional, WeightsPrecisions_integer, ActivationPrecisions_integer, InputPrecisions_integer)

    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))

