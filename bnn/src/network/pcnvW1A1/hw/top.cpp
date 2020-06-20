/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file top.cpp
 *
 * HLS Description of the CNV BNN with axi-lite based parameter loading (DoMemInit) 
 * and  dataflow architecture of the image inference (DoCompute).
 * The network uses 1 bit weights and 1 bit activation.
 *
 *****************************************************************************/
#include "config.h"

#include "bnn-library.h"

#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;


/* There are more binary weights than activations since passthrough activation is used at the output layer. */
static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) {
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
      // do nothing, no thres mem for layer 3 as PassThrough activation is used
      break;
  }
}
void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {

/*Stream declaration.*/

#pragma HLS DATAFLOW
  /*3 streams to pack data from 64 bits to 24.*/
  stream<ap_uint<64>> inter0("DoCompute.inter0");
  stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
  stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 depth=128
  stream<ap_uint<24>> inter0_2_padded("DoCompute.inter0_2_padded");/*IN CNV-LAYER 0*/
//#pragma HLS STREAM variable=inter0_2_padded depth=128

  stream<ap_uint<L0_OFM_CH>> inter1("DoCompute.inter1");/*OUT L0-IN MAX POOL 0*/
  stream<ap_uint<L0_OFM_CH>> inter2("DoCompute.inter2");
#pragma HLS STREAM variable=inter2 depth=128
  stream<ap_uint<L0_OFM_CH>> inter2_padded("DoCompute.inter2_padded");/*IN CNV-LAYER 1*/
//#pragma HLS STREAM variable=inter2_padded depth=128

  stream<ap_uint<L1_OFM_CH>> inter3("DoCompute.inter3"); /*OUT L1- IN MAX POOL 1*/
  stream<ap_uint<L1_OFM_CH>> inter4("DoCompute.inter4");
#pragma HLS STREAM variable=inter4 depth=128
  stream<ap_uint<L1_OFM_CH>> inter4_padded("DoCompute.inter4_padded");/*IN CNV-LAYER 2*/
//#pragma HLS STREAM variable=inter4_padded depth=128

  stream<ap_uint<L2_OFM_CH>> inter5("DoCompute.inter5"); /*IN MAX POOL 2*/
#pragma HLS STREAM variable=inter5 depth=128
  stream<ap_uint<L2_OFM_CH>> inter6("DoCompute.inter6");/*OUT MAXPOOL SIZE: 4x4x64*/
  stream<ap_uint<512>> inter6_1("DoCompute.inter6_flattened");/*IN FC-LAYER 3*/
#pragma HLS STREAM variable=inter6 depth=128
  stream<ap_uint<64>> memOutStrm("DoCompute.memOutStrm");


  const unsigned int inBits = 32 * 32 * 3 * 8;
  // const unsigned int inBitsPadded = paddedSize(inBits, 64);
  /* Since output is the accumulator value it is 16bits(accum width) time the output vector.*/
  const unsigned int outBits = L3_MH*16;

    /*Data in packing in 64 bit chunks (192 bits is multiple of 24bit and 64bit )*/
  Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);
  StreamingDataWidthConverter_Batch<64, 192, (32 * 32 * 3 * 8) / 64>(inter0, inter0_1, numReps);
  StreamingDataWidthConverter_Batch<192, 24, (32 * 32 * 3 * 8) / 192>(inter0_1, inter0_2, numReps);
  
  
  // convolutional layers
  
  
  FMPadding_Batch<L0_IFM_DIM,     // Input size = 32
                  L0_IFM_DIM_PAD, // Output size = 34
                  2,              // Padding = 2, so pad left = pad right = pad up = pad down = 1 extra pixel each side.
                  L0_IFM_CH,      // Input channels = 3
                  3,              // SIMD = 3 because we are processing 24 bit streams.
                  ap_uint<8>,     // The input is represented by RGB ubyte pixels
                  0>              // Padding style = 0 (choose 0 so that no extra padding is applied)
                  (inter0_2,inter0_2_padded,numReps);
  
  ConvLayer_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM_PAD, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Slice<ap_uint<1>,1>, Recast<Binary>>
    (inter0_2_padded, inter1, weights0, threshs0, numReps, ap_resource_lut());
  StreamingMaxPool_Batch<L0_OFM_DIM, 2, L0_OFM_CH>(inter1, inter2, numReps);

  FMPadding_Batch<L1_IFM_DIM,     // Input size = 32
                  L1_IFM_DIM_PAD, // Output size = 18
                  2,              // Padding = 2
                  L1_IFM_CH,      // 16 channels input
                  L0_OFM_CH,      // SIMD = 16 because we are packing 1 bit activations in 16 bit streams
                  ap_uint<1>,     // At this stage activations are 1 bit
                  0>              // Padding style = 0
                  (inter2,inter2_padded,numReps);
  ConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM_PAD, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Recast<XnorMul>, Slice<ap_uint<1>>>
    (inter2_padded, inter3, weights1, threshs1, numReps, ap_resource_lut());
  StreamingMaxPool_Batch<L1_OFM_DIM, 2, L1_OFM_CH>(inter3, inter4, numReps);

  FMPadding_Batch<L2_IFM_DIM,
                  L2_IFM_DIM_PAD,
                  2,
                  L2_IFM_CH,
                  L1_OFM_CH,
                  ap_uint<1>,
                  0>(inter4,inter4_padded,numReps);
  ConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM_PAD, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Recast<XnorMul>, Slice<ap_uint<1>>>
    (inter4_padded, inter5, weights2, threshs2, numReps, ap_resource_lut());
  StreamingMaxPool_Batch<L2_OFM_DIM, 2, L2_OFM_CH>(inter5, inter6, numReps);
  
  //Differently from previous cnv designs, the output is not reduced to 1x1xN_CHANNELS in this case.
  // Output is 4x4xN_CHANNELS. We may need to flatten the vector this time.
  
  StreamingDataWidthConverter_Batch<L2_OFM_CH,                           //Input width
                                    512,                                 //Output width
                                    (4 * 4 * L2_OFM_CH * 1) / L2_OFM_CH> // TOTAL BITS / Input width
                                    (inter6, inter6_1, numReps);
  // fully connected layers
  
  
  WidthAdjustedOutputStream<16 * L3_PE, 64, L3_MH / L3_PE>  wa_out(memOutStrm, numReps);

  StreamingFCLayer_Batch<L3_MW, L3_MH, L3_SIMD, L3_PE, Recast<XnorMul>, Slice<ap_uint<16> >>
    (inter6_1, static_cast<hls::stream<ap_uint<16 * L3_PE>>&>(wa_out), weights3, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

  Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);

}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
