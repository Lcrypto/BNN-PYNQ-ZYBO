/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    32  IFM_CH =     3
 *      OFM  =    28  OFM_CH =    16
 *     SIMD  =     3    PE   =     8
 *     WMEM  =    50   TMEM  =     2
 *     #Ops  = 1881600   Ext Latency  = 39200
**/

#define L0_K 5
#define L0_IFM_CH 3
#define L0_IFM_DIM 32
#define L0_OFM_CH 16
#define L0_OFM_DIM 28
#define L0_SIMD 3
#define L0_PE 8
#define L0_WMEM 50
#define L0_TMEM 2
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    10  OFM_CH =    32
 *     SIMD  =     8    PE   =     8
 *     WMEM  =   200   TMEM  =     4
 *     #Ops  = 2560000   Ext Latency  = 20000
**/

#define L1_K 5
#define L1_IFM_CH 16
#define L1_IFM_DIM 14
#define L1_OFM_CH 32
#define L1_OFM_DIM 10
#define L1_SIMD 8
#define L1_PE 8
#define L1_WMEM 200
#define L1_TMEM 4
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =     5  IFM_CH =    32
 *      OFM  =     1  OFM_CH =    64
 *     SIMD  =     2    PE   =     2
 *     WMEM  = 12800   TMEM  =    32
 *     #Ops  = 102400   Ext Latency  = 12800
**/

#define L2_K 5
#define L2_IFM_CH 32
#define L2_IFM_DIM 5
#define L2_OFM_CH 64
#define L2_OFM_DIM 1
#define L2_SIMD 2
#define L2_PE 2
#define L2_WMEM 12800
#define L2_TMEM 32
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =    64 MatH =   128
 *     SIMD =     1  PE  =     1
 *     WMEM =  8192 TMEM =   128
 *     #Ops  = 16384   Ext Latency  =  8192
**/

#define L3_SIMD 1
#define L3_PE 1
#define L3_WMEM 8192
#define L3_TMEM 128
#define L3_MW 64
#define L3_MH 128
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Fully-Connected Layer L4:
 *     MatW =   128 MatH =   128
 *     SIMD =     2  PE  =     2
 *     WMEM =  4096 TMEM =    64
 *     #Ops  = 32768   Ext Latency  =  4096
**/

#define L4_SIMD 2
#define L4_PE 2
#define L4_WMEM 4096
#define L4_TMEM 64
#define L4_MW 128
#define L4_MH 128
#define L4_WPI 1
#define L4_API 1
#define L4_WPF 0
#define L4_APF 0

/**
 * Fully-Connected Layer L5:
 *     MatW =   128 MatH =    64
 *     SIMD =     1  PE  =     4
 *     WMEM =  2048 TMEM =    16
 *     #Ops  = 16384   Ext Latency  =  2048
**/

#define L5_SIMD 1
#define L5_PE 4
#define L5_WMEM 2048
#define L5_TMEM 16
#define L5_MW 128
#define L5_MH 64
#define L5_WPI 1
#define L5_API 16
#define L5_WPF 0
#define L5_APF 0

#endif //__LAYER_CONFIG_H_
