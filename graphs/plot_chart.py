import matplotlib.pyplot as plt
import matplotlib
import numpy as np

Bytes               = [8, 16, 32, 64, 128, 256]
MA_thread           = [409000000,225000000,139000000,73000000,41000000,22000000]
MA_warp             = [385000000,226000000,128000000, 73000000, 42000000, 22000000]
RS_block_thread     = [138855632,84832040,114003816,103395904,116966040,98282304]
RS_non_block_thread = [90023752,104924216, 78158480, 91377048,103159456,102703920]
RS_block_warp       = [126298072,110371520,115624752,128524040,109555096,119712696]
RS_non_block_warp   = [112542680,127405912,127417520, 79255784,109958072,139597616]

x_min = 10**8
x_min = min([x_min], MA_thread)
x_min = min(x_min, MA_warp)
x_min = min(x_min, RS_block_thread)
x_min = min(x_min, RS_non_block_thread)
x_min = min(x_min, RS_block_warp)
x_min = min(x_min, RS_non_block_warp)
x_min = min(x_min)

x_max = 10**8
x_max = max([x_max], MA_thread)
x_max = max(x_max, MA_warp)
x_max = max(x_max, RS_block_thread)
x_max = max(x_max, RS_non_block_thread)
x_max = max(x_max, RS_block_warp)
x_max = max(x_max, RS_non_block_warp)
x_max = max(x_max)
print("x_min", x_min, "x_max", x_max)

plt.plot(Bytes, MA_thread, '-', label = 'MA: one blocking request per thread')
plt.plot(Bytes, MA_warp, '--', label = 'MA: one blocking request per warp')
plt.plot(Bytes, RS_block_thread, '-', label = 'RS: one blocking request per thread')
plt.plot(Bytes, RS_non_block_thread, '-', label = 'RS: one non-blocking request per thread')
plt.plot(Bytes, RS_block_warp, '--', label = 'RS: one blocking request per warp')
plt.plot(Bytes, RS_non_block_warp, '--', label = 'RS: one non-blocking request per warp')

plt.xlabel("Allocation size in bytes")
plt.ylabel("Number of requests per second in millions")
plt.legend()
plt.title('Performance Improvement: Runtime System vs. Monolithic Applicaiton')
plt.yscale('log') 
#plt.yticks([10**7, 10**8, 10**9])


#plt.xticks(np.arange(x_min, x_max+1, 1.0))
#plt.yticks(np.logspace(x_min, x_max+1, num=6))

plt.savefig('performance_results.pdf')
plt.show()
