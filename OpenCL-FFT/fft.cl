#if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

//#define INVERSE

inline double2 cplx_mul(double2 lhs, double2 rhs) {
    return (double2) {
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

double2 calculate_omega(double alpha) {
    double r;
    double i = sincos(alpha, &r);
    return (double2) {r, i};
}

kernel void fft(global double2* data, local double2* local_cache, local double2* omega_cache, int N) 
{
    const int local_size = get_local_size(0);
    
    const int global_id = get_global_id(0);

    #ifdef INVERSE
        const double angle = 2.0 * M_PI / N;
    #else
        const double angle = -2.0 * M_PI / N;
    #endif

    const int g_offset = N / 2 / local_size;

    const int btid = global_id * g_offset;

    for(int i = btid; i < btid + g_offset; i++)
    {
        omega_cache[i] = calculate_omega(i * angle);
    }
    
    const int leading_zeroes = clz(N) + 1;
    const int logTwo = 32 - leading_zeroes;
    
    for(int i = btid * 2; i < btid * 2 + g_offset * 2; i++)
    {
        int j = rev_bits(i) >> leading_zeroes;
        local_cache[i] = data[j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < logTwo; i++)
    {
    
        for(int j = btid; j < btid + g_offset; j++)
        {
            const unsigned even = index_map(j, i, N);
            const unsigned odd = even + (1u << i);
            
            const int q = (j & (N / (1 << (logTwo - i)) - 1)) * (1 << (logTwo - i)) / 2;

            const double2 e = cplx_mul(omega_cache[q], local_cache[odd]);
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            local_cache[odd] = local_cache[even] - e;
            local_cache[even] += e;
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int i = btid * 2; i < btid * 2 + g_offset * 2; i++)
    {
    #ifdef INVERSE
        data[i] = local_cache[i] / N;
    #else
        data[i] = local_cache[i];
    #endif
    }
    
}
