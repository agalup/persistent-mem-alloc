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

fig1, ax1 = plt.subplots()
ax1.plot(Bytes, MA_thread, '-', label = 'MA: one blocking request per thread')
ax1.plot(Bytes, MA_warp, '--', label = 'MA: one blocking request per warp')
ax1.plot(Bytes, RS_block_thread, '-', label = 'RS: one blocking request per thread')
ax1.plot(Bytes, RS_non_block_thread, '-', label = 'RS: one non-blocking request per thread')
ax1.plot(Bytes, RS_block_warp, '--', label = 'RS: one blocking request per warp')
ax1.plot(Bytes, RS_non_block_warp, '--', label = 'RS: one non-blocking request per warp')

ax1.set_xlabel("Allocation size in bytes per thread")
ax1.set_ylabel("Number of requests per second")
ax1.legend()
ax1.set_title('Performance Improvement: Runtime System vs. Monolithic Applicaiton', y=1.08)

ax1.set_yscale('log')
ax1.set_yticks([0.2*(10**8), 0.4*(10**8), 0.6*(10**8), 0.8*(10**8), 10**8, 2.0*(10**8), 4.0*(10**8)])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.savefig('performance_results.pdf')
plt.show()
