#!/usr/bin/env python3
"""Test GPU utilization with multiprocessing"""

import multiprocessing as mp
import time
import os

def gpu_work_process(process_id):
    """Simulate GPU work in a separate process"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            pid = os.getpid()
            print(f'Process {process_id} (PID {pid}): Starting GPU work on {device}')
            
            # Create some GPU work
            for i in range(100):
                # Create large tensors and do matrix multiplication
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()  # Force completion
                time.sleep(0.01)  # Small delay
            
            print(f'Process {process_id} (PID {pid}): Finished GPU work')
        else:
            print(f'Process {process_id}: No CUDA available')
    except Exception as e:
        print(f'Process {process_id}: Error: {e}')

if __name__ == '__main__':
    print('🧪 Testing multiprocessing GPU utilization WITHOUT MPS...')
    
    # Set spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    print('Starting 4 processes doing GPU work...')
    
    start_time = time.time()
    
    processes = []
    for i in range(4):
        p = mp.Process(target=gpu_work_process, args=(i,))
        processes.append(p)
        p.start()
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f'Total time with multiprocessing: {end_time - start_time:.2f} seconds')
    print('Compare this to threading times above')
