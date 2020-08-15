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
    return ((threadId & (N - (1u << currentIteration))) << 1) | (threadId & ((1u << currentIteration) - 1));
}

inline unsigned twiddle_map(unsigned threadId, unsigned currentIteration, unsigned logTwo, unsigned N)
{
    return (threadId & (N / (1u << (logTwo - currentIteration)) - 1)) * (1u << (logTwo - currentIteration)) / 2;
}

float2 twiddle(float kn, float N, bool is_inverse) {
    float r;
    float i = sincos((!is_inverse * 2 - 1) * 2.0 * M_PI * kn / N, &r);

    return (float2) {r, i};
}

inline void dft2(local float2* local_cache, const int threadId, const int currentIteration, const int logTwo, const int N, bool is_inverse)
{
    const unsigned even = index_map(threadId, currentIteration, N);
    const unsigned odd = even + (1u << currentIteration);
    
    float2 evenVal = local_cache[even];
    const unsigned q = twiddle_map(threadId, currentIteration, logTwo, N);

    const float2 e = cplx_mul(twiddle(q, N, is_inverse), local_cache[odd]);
            
    local_cache[odd] = local_cache[even] - e;
    local_cache[even] += e;
}

kernel void fft(global float2* data, local float2* local_cache, const int N, const int radix, const int is_inverse) 
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    const int g_offset = N / radix / local_size;

    const int btid = global_id * g_offset;
    
    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;
    
    for(int i = btid * radix; i < btid * radix + g_offset * radix; i++)
    {
        const int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    const int half_radix = radix / 2;

    for(int i = 0; i < logTwo; i++)
    {
        for(int j = btid; j < btid + g_offset; j++)
        {
            const int t = j * half_radix;
            
            for(int r = 0; r < half_radix; r++)
                dft2(local_cache, t + r, i, logTwo, N, is_inverse);
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    for(int i = btid * radix; i < btid * radix + g_offset * radix; i++)
    {
        data[i] = local_cache[i] / (N * is_inverse + 1 * !is_inverse);
    }
    
}
