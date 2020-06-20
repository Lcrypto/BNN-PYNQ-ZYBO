/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    34  IFM_CH =     3
 *      OFM  =    32  OFM_CH =     8
 *     SIMD  =     3    PE   =     8
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 442368   Ext Latency  =  9216
**/

#define L0_K 3
#define L0_IFM_CH 3
#define L0_IFM_DIM_PAD 34
#define L0_IFM_DIM 32
#define L0_OFM_CH 8
#define L0_OFM_DIM 32
#define L0_SIMD 3
#define L0_PE 8
#define L0_WMEM 9
#define L0_TMEM 1
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    18  IFM_CH =     8
 *      OFM  =    16  OFM_CH =    16
 *     SIMD  =     8    PE   =     8
 *     WMEM  =    18   TMEM  =     2
 *     #Ops  = 589824   Ext Latency  =  4608
**/

#define L1_K 3
#define L1_IFM_CH 8
#define L1_IFM_DIM_PAD 18
#define L1_IFM_DIM 16
#define L1_OFM_CH 16
#define L1_OFM_DIM 16
#define L1_SIMD 8
#define L1_PE 8
#define L1_WMEM 18
#define L1_TMEM 2
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =    10  IFM_CH =    16
 *      OFM  =     8  OFM_CH =    32
 *     SIMD  =     2    PE   =     2
 *     WMEM  =  1152   TMEM  =    16
 *     #Ops  = 589824   Ext Latency  = 73728
**/

#define L2_K 3
#define L2_IFM_CH 16
#define L2_IFM_DIM_PAD 10
#define L2_IFM_DIM 8
#define L2_OFM_CH 32
#define L2_OFM_DIM 8
#define L2_SIMD 2
#define L2_PE 2
#define L2_WMEM 1152
#define L2_TMEM 16
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =   512 MatH =    64
 *     SIMD =     1  PE  =     4
 *     WMEM =  8192 TMEM =    16
 *     #Ops  = 65536   Ext Latency  =  8192
**/

#define L3_SIMD 1
#define L3_PE 4
#define L3_WMEM 8192
#define L3_TMEM 16
#define L3_MW 512
#define L3_MH 64
#define L3_WPI 1
#define L3_API 16
#define L3_WPF 0
#define L3_APF 0

#endif //__LAYER_CONFIG_H_
