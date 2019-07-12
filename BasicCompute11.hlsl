//--------------------------------------------------------------------------------------
// File: BasicCompute11.hlsl
//
// This file contains the Compute Shader to perform array A + array B
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "IntelExtensions.hlsl"

StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

static int VEC_SIZE = 1;
static int TILE_M = 4;
static int TILE_K = 16;
static int TILE_N = 8;

static uint3 gl_WorkGroupID = uint3(0, 0, 0);
static uint3 gl_LocalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_WorkGroupID : SV_GroupID;
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
};


void initGLBuiltins(CS_INPUT input)
{
    gl_WorkGroupID = input.dx_WorkGroupID;
    gl_LocalInvocationID = input.dx_LocalInvocationID;
};

[numthreads(8, 1, 1)]
void L3_SIMD_4x1_1x8(CS_INPUT input)
{
    IntelExt_Init();
    initGLBuiltins(input);
    const int M = 512, N = M, K = M;
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);

    // Result ctile is M rows x N columns
    // M = 4, we have 1 row of work-items, so we need 4/1 = 4 results down
    // N = 8, we have 8 columns of work-items, so we need 8/8 = 1 result across

    float   dot00 = 0.f;
    float   dot01 = 0.f;
    float   dot02 = 0.f;
    float   dot03 = 0.f;

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( group_y * TILE_M ) * width1;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( group_y * TILE_M ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        // We want to load atile, which is M rows x K columns
        // M = 4, we have 1 row of work-items, so each work-item must load 4/1 = 4 rows
        // K = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        float   arow;

        // Now load btile, which is K rows x N columns
        // K = 8, we have 1 row of work-items, so each work-item must load 8/1 = 8 rows
        // N = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        float  brow0 = src1[src1_read0];  src1_read0 += width1;
        float  brow1 = src1[src1_read0];  src1_read0 += width1;
        float  brow2 = src1[src1_read0];  src1_read0 += width1;
        float  brow3 = src1[src1_read0];  src1_read0 += width1;
        float  brow4 = src1[src1_read0];  src1_read0 += width1;
        float  brow5 = src1[src1_read0];  src1_read0 += width1;
        float  brow6 = src1[src1_read0];  src1_read0 += width1;
        float  brow7 = src1[src1_read0];  src1_read0 += width1;

        arow = src0[src0_read + 0 * width0 ];
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 0 ), brow0, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 1 ), brow1, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 2 ), brow2, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 3 ), brow3, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 4 ), brow4, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 5 ), brow5, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 6 ), brow6, dot00);
        dot00 = mad(IntelExt_WaveReadLaneAt( arow, 7 ), brow7, dot00);

        arow = src0[src0_read + 1 * width0 ];
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 0 ), brow0, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 1 ), brow1, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 2 ), brow2, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 3 ), brow3, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 5 ), brow5, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 4 ), brow4, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 6 ), brow6, dot01);
        dot01 = mad(IntelExt_WaveReadLaneAt( arow, 7 ), brow7, dot01);

        arow = src0[src0_read + 2 * width0 ];
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 0 ), brow0, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 1 ), brow1, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 2 ), brow2, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 3 ), brow3, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 4 ), brow4, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 5 ), brow5, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 6 ), brow6, dot02);
        dot02 = mad(IntelExt_WaveReadLaneAt( arow, 7 ), brow7, dot02);

        arow = src0[src0_read + 3 * width0 ];
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 0 ), brow0, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 1 ), brow1, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 2 ), brow2, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 3 ), brow3, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 4 ), brow4, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 5 ), brow5, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 6 ), brow6, dot03);
        dot03 = mad(IntelExt_WaveReadLaneAt( arow, 7 ), brow7, dot03);

        src0_read += TILE_K / VEC_SIZE;
        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    dst[dst_write0] = dot00;  dst_write0 += width1;
    dst[dst_write0] = dot01;  dst_write0 += width1;
    dst[dst_write0] = dot02;  dst_write0 += width1;
    dst[dst_write0] = dot03;  dst_write0 += width1;
}
