import pandas as pd
import numpy as np
import argparse
import sys

def process_trace(gpu_csv, nvtx_csv, target_tag, output_csv):
    print(f"Loading NVTX trace: {nvtx_csv}")
    nvtx_df = pd.read_csv(nvtx_csv)
    
    # 1. Find the target NVTX tag window
    tag_data = nvtx_df[nvtx_df['Name'] == target_tag]
    
    if tag_data.empty:
        print(f"Error: Tag '{target_tag}' not found in {nvtx_csv}")
        print("Available tags include:")
        print(nvtx_df['Name'].unique()[:10])
        return

    # Use the first occurrence if multiple exist
    tag_start = tag_data['Start (ns)'].iloc[0]
    tag_end = tag_data['End (ns)'].iloc[0]
    tag_duration_ms = (tag_end - tag_start) / 1_000_000
    
    print(f"Found tag: {target_tag}")
    print(f"Window: {tag_start} ns to {tag_end} ns ({tag_duration_ms:.2f} ms)")

    print(f"Loading GPU trace: {gpu_csv}")
    gpu_df = pd.read_csv(gpu_csv)
    gpu_df['End (ns)'] = gpu_df['Start (ns)'] + gpu_df['Duration (ns)']

    # 2. Filter GPU activity within the NVTX window
    mask = (gpu_df['Start (ns)'] < tag_end) & (gpu_df['End (ns)'] > tag_start)
    active_df = gpu_df[mask].copy()

    # 3. Define 1ms bins
    MS_IN_NS = 1_000_000
    bins = np.arange(tag_start, tag_end + MS_IN_NS, MS_IN_NS)
    
    results = []

    print(f"Processing {len(bins)-1} ms intervals...")
    for i in range(len(bins) - 1):
        b_start, b_end = bins[i], bins[i+1]
        
        # Find ops in this 1ms bin
        bin_ops = active_df[(active_df['Start (ns)'] < b_end) & (active_df['End (ns)'] > b_start)].copy()
        
        if bin_ops.empty:
            results.append({
                'timestamp_ms': i,
                'compute_pct': 0.0,
                'memcpy_pct': 0.0,
                'total_gpu_pct': 0.0,
                'throughput_mb_s': 0.0
            })
            continue

        # Calculate clipped durations within the bin
        bin_ops['clipped_dur'] = bin_ops[['End (ns)', 'End (ns)']].min(axis=1).clip(upper=b_end) - \
                                 bin_ops[['Start (ns)', 'Start (ns)']].max(axis=1).clip(lower=b_start)
        
        is_memcpy = bin_ops['Name'].str.contains('memcpy', case=False, na=False)
        
        # Compute and Memcpy Percentages
        comp_ns = bin_ops[~is_memcpy]['clipped_dur'].sum()
        mem_ns = bin_ops[is_memcpy]['clipped_dur'].sum()
        
        comp_pct = min(comp_ns / MS_IN_NS, 1.0) * 100
        mem_pct = min(mem_ns / MS_IN_NS, 1.0) * 100
        
        # Weighted Throughput Calculation (Avg MB/s during the transfer time)
        memcpy_ops = bin_ops[is_memcpy]
        if not memcpy_ops.empty and mem_ns > 0:
            # Weight throughput by how much of the 1ms bin each transfer occupied
            weighted_tp = (memcpy_ops['Throughput (MB/s)'] * memcpy_ops['clipped_dur']).sum() / mem_ns
        else:
            weighted_tp = 0.0

        results.append({
            'timestamp_ms': i,
            'compute_pct': round(comp_pct, 4),
            'memcpy_pct': round(mem_pct, 4),
            'total_gpu_pct': round(min(comp_pct + mem_pct, 100.0), 4),
            'throughput_mb_s': round(weighted_tp, 2)
        })

    # 4. Save results
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_csv, index=False)
    print(f"Successfully saved 1ms analysis to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze 1ms GPU usage within a specific NVTX tag.")
    parser.add_argument("--gpu", required=True, help="Input cuda_gpu_trace.csv")
    parser.add_argument("--nvtx", required=True, help="Input nvtx_pushpop_trace.csv")
    parser.add_argument("--tag", default=":alpamayo_clip:48ec6a11-5802-4c02-9912-c0fe78937110", help="Target NVTX Tag Name")
    parser.add_argument("--output", default="clip_usage_1ms.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    process_trace(args.gpu, args.nvtx, args.tag, args.output)
