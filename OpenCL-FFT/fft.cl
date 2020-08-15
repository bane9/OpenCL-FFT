//#define INVERSE
//#define RADIX_2
//#define RADIX_4
//#define RADIX_8
//#define RADIX_16
#define RADIX_32

inline float2 cplx_mul(float2 lhs, float2 rhs) {
    return (float2) {
        lhs.even * rhs.even - lhs.odd * rhs.odd,
        lhs.odd * rhs.even + lhs.even * rhs.odd
    };
}

unsigned int rev_bits(unsigned int num) 
{ 
    unsigned int count = sizeof(num) * 8 - 1; 
    unsigned int reverse_num = num; 
      
    num >>= 1;  
    while(num) 
    { 
       reverse_num <<= 1;        
       reverse_num |= num & 1; 
       num >>= 1; 
       count--; 
    } 
    reverse_num <<= count; 
    return reverse_num; 
} 

inline unsigned index_map(unsigned threadId, unsigned currentIteration, unsigned N){
    return ((threadId & (N - (1u << currentIteration))) << 1u) | (threadId & ((1u << currentIteration) - 1u));
}

inline unsigned twiddle_map(unsigned threadId, unsigned currentIteration, unsigned logTwo, unsigned N)
{
    return (threadId & (N / (1u << (logTwo - currentIteration)) - 1)) * (1u << (logTwo - currentIteration)) / 2;
}

float2 twiddle(float kn, float N) {
    float r;
    #ifdef INVERSE
        float i = sincos(2.0 * M_PI * kn / N, &r);
    #else
        float i = sincos(-2.0 * M_PI * kn / N, &r);
    #endif

    return (float2) {r, i};
}

inline void dft2(local float2* local_cache, const int threadId, const int currentIteration, const int logTwo, const int N)
{
    const unsigned even = index_map(threadId, currentIteration, N);
    const unsigned odd = even + (1u << currentIteration);
            
    const unsigned q = twiddle_map(threadId, currentIteration, logTwo, N);

    const float2 e = cplx_mul(twiddle(q, N), local_cache[odd]);
            
    local_cache[odd] = local_cache[even] - e;
    local_cache[even] += e;
}

#if defined RADIX_2
kernel void fft(global float2* data, local float2* local_cache, const int N) 
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int g_offset = N / 2 / local_size;

    const int btid = global_id * g_offset;
    
    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;
    
    for(int i = btid * 2; i < btid * 2 + g_offset * 2; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            dft2(local_cache, j, i, logTwo, N);
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    for(int i = btid * 2; i < btid * 2 + g_offset * 2; i++)
    {
    #ifdef INVERSE
        data[i] = local_cache[i] / N;
    #else
        data[i] = local_cache[i];
    #endif
    }
    
}
#elif defined RADIX_4
kernel void fft(global float2* data, local float2* local_cache, const int N)
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;

    const int g_offset = N / 4 / local_size;

    const int btid = global_id * g_offset;

    for(int i = btid * 4; i < btid * 4 + g_offset * 4; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            const int t = j * 2; 

            dft2(local_cache, t, i, logTwo, N);
            dft2(local_cache, t + 1, i, logTwo, N);
                
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(int i = btid * 4; i < btid * 4 + g_offset * 4; i++)
    {
        #ifdef INVERSE
            data[i] = local_cache[i] / N;
        #else
            data[i] = local_cache[i];
        #endif
    }
}
#elif defined RADIX_8
kernel void fft(global float2* data, local float2* local_cache, const int N)
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;

    const int g_offset = N / 8 / local_size;

    const int btid = global_id * g_offset;

    for(int i = btid * 8; i < btid * 8 + g_offset * 8; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            const int t = j * 2;

            dft2(local_cache, t, i, logTwo, N);
            dft2(local_cache, t + 1, i, logTwo, N);
                
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(int i = btid * 8; i < btid * 8 + g_offset * 8; i++)
    {
        #ifdef INVERSE
            data[i] = local_cache[i] / N;
        #else
            data[i] = local_cache[i];
        #endif
    }
}
#elif defined RADIX_16
kernel void fft(global float2* data, local float2* local_cache, const int N)
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;

    const int g_offset = N / 16 / local_size;

    const int btid = global_id * g_offset;

    for(int i = btid * 16; i < btid * 16 + g_offset * 16; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            const int t = j * 8; 

            dft2(local_cache, t, i, logTwo, N);
            dft2(local_cache, t + 1, i, logTwo, N);
            dft2(local_cache, t + 2, i, logTwo, N);
            dft2(local_cache, t + 3, i, logTwo, N);
            dft2(local_cache, t + 4, i, logTwo, N);
            dft2(local_cache, t + 5, i, logTwo, N);
            dft2(local_cache, t + 6, i, logTwo, N);
            dft2(local_cache, t + 7, i, logTwo, N);
                
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(int i = btid * 16; i < btid * 16 + g_offset * 16; i++)
    {
        #ifdef INVERSE
            data[i] = local_cache[i] / N;
        #else
            data[i] = local_cache[i];
        #endif
    }
}
#elif defined RADIX_32
kernel void fft(global float2* data, local float2* local_cache, const int N)
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;

    const int g_offset = N / 32 / local_size;

    const int btid = global_id * g_offset;

    for(int i = btid * 32; i < btid * 32 + g_offset * 32; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            const int t = j * 16; 

            dft2(local_cache, t, i, logTwo, N);
            dft2(local_cache, t + 1, i, logTwo, N);
            dft2(local_cache, t + 2, i, logTwo, N);
            dft2(local_cache, t + 3, i, logTwo, N);
            dft2(local_cache, t + 4, i, logTwo, N);
            dft2(local_cache, t + 5, i, logTwo, N);
            dft2(local_cache, t + 6, i, logTwo, N);
            dft2(local_cache, t + 7, i, logTwo, N);
            dft2(local_cache, t + 8, i, logTwo, N);
            dft2(local_cache, t + 9, i, logTwo, N);
            dft2(local_cache, t + 10, i, logTwo, N);
            dft2(local_cache, t + 11, i, logTwo, N);
            dft2(local_cache, t + 12, i, logTwo, N);
            dft2(local_cache, t + 13, i, logTwo, N);
            dft2(local_cache, t + 14, i, logTwo, N);
            dft2(local_cache, t + 15, i, logTwo, N);
                
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(int i = btid * 32; i < btid * 32 + g_offset * 32; i++)
    {
        #ifdef INVERSE
            data[i] = local_cache[i] / N;
        #else
            data[i] = local_cache[i];
        #endif
    }
}
#endif
