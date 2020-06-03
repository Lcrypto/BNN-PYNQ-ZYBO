/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Fully-Connected Layer L0:
 *     MatW =   832 MatH =   256
 *     SIMD =    32  PE  =    32
 *     WMEM =   208 TMEM =     8
 *     #Ops  = 425984   Ext Latency  =   208
**/

#define L0_SIMD 32
#define L0_PE 32
#define L0_WMEM 208
#define L0_TMEM 8
#define L0_MW 832
#define L0_MH 256
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Fully-Connected Layer L1:
 *     MatW =   256 MatH =   256
 *     SIMD =    16  PE  =    16
 *     WMEM =   256 TMEM =    16
 *     #Ops  = 131072   Ext Latency  =   256
**/

#define L1_SIMD 16
#define L1_PE 16
#define L1_WMEM 256
#define L1_TMEM 16
#define L1_MW 256
#define L1_MH 256
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Fully-Connected Layer L2:
 *     MatW =   256 MatH =   256
 *     SIMD =    16  PE  =    16
 *     WMEM =   256 TMEM =    16
 *     #Ops  = 131072   Ext Latency  =   256
**/

#define L2_SIMD 16
#define L2_PE 16
#define L2_WMEM 256
#define L2_TMEM 16
#define L2_MW 256
#define L2_MH 256
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =   256 MatH =    64
 *     SIMD =     8  PE  =     8
 *     WMEM =   256 TMEM =     8
 *     #Ops  = 32768   Ext Latency  =   256
**/

#define L3_SIMD 8
#define L3_PE 8
#define L3_WMEM 256
#define L3_TMEM 8
#define L3_MW 256
#define L3_MH 64
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

#endif //__LAYER_CONFIG_H_
