#!/usr/bin/env python3
import argparse
import os
import re
import csv
from collections import defaultdict

RULES = [
    # Attention
    (re.compile(r"fmha|flash[_:]|pytorch_flash::flash_|MemEffAttention|cutlass.*Attention", re.I), "aten::attention"),

    # GEMM / MatMul / Linear
    (re.compile(r"\bnvjet_tst_", re.I), "aten::matmul/mm/bmm/linear"),
    (re.compile(r"cublas.*gemm|gemmex|GemmEx|cutlass::gemm|wgmma|GEMM", re.I), "aten::matmul/mm/bmm/linear"),
    (re.compile(r"\bgemv\b|gemv2|cublasGemv", re.I), "aten::matmul/mm/bmm/linear"),
    (re.compile(r"cublasLt::splitKreduce", re.I), "aten::matmul/mm/bmm/linear"),

    # Cat / concat
    (re.compile(r"\bCatArrayBatchedCopy\b|CatArrayBatchedCopy_aligned|CatArr", re.I), "aten::cat"),

    # Norm
    (re.compile(r"layer[_ ]?norm|vectorized_layer_norm_kernel|rmsnorm|rms[_ ]?norm", re.I), "aten::norm"),

    # Softmax
    (re.compile(r"SoftMaxForward|softmax", re.I), "aten::softmax"),

    # Reductions
    (re.compile(r"\breduce_kernel\b|ReduceOp|BlockReduce|MeanOps|\bsum\b", re.I), "aten::reduce"),

    # Copies / casts / mem
    (re.compile(r"memcpy|Memcpy|HtoD|DtoH|DtoD", re.I), "cuda::memcpy"),
    (re.compile(r"direct_copy_kernel_cuda|copy_kernel|CopyKernel|\bcopy\b", re.I), "aten::copy_"),
    (re.compile(r"bfloat16_copy_kernel_cuda|half_copy_kernel_cuda|cast|Convert|to_copy", re.I), "aten::to/_to_copy"),

    # Elementwise functors (needs full kernel name to see these!)
    (re.compile(r"CUDAFunctor_add\b|CUDAFunctorOnSelf_add\b", re.I), "aten::add"),
    (re.compile(r"binary_internal::MulFunctor|MulFunctor\b", re.I), "aten::mul"),
    (re.compile(r"binary_internal::DivFunctor|DivFunctor\b", re.I), "aten::div"),
    (re.compile(r"binary_internal::SubFunctor|SubFunctor\b", re.I), "aten::sub"),
    (re.compile(r"CompareEqFunctor|CompareEQ", re.I), "aten::eq"),
    (re.compile(r"FillFunctor\b", re.I), "aten::fill_"),
    (re.compile(r"pow_tensor_scalar_kernel_impl|\bpow\b", re.I), "aten::pow"),

    # Indexing / gather/scatter
    (re.compile(r"index_elementwise_kernel|write_indices|gather_kernel", re.I), "aten::index/gather"),

    # CUB algorithms (sort/scan/select) – these typically back indexing/unique/topk etc.
    (re.compile(r"\bat_cuda_detail::cub::detail::radix_sort", re.I), "aten::sort"),
    (re.compile(r"\bat_cuda_detail::cub::detail::scan", re.I), "aten::scan"),
    (re.compile(r"\bat_cuda_detail::cub::detail::select", re.I), "aten::select"),

    # --- Activations / math kernels that encode op name directly
    (re.compile(r"\bGeluCUDAKernelImpl\b|\bGeluType\b|\bgelu\b", re.I), "aten::gelu"),
    (re.compile(r"\bsilu_kernel\b|\bsilu\b", re.I), "aten::silu"),
    (re.compile(r"\brsqrt_kernel_cuda\b|\brsqrt\b", re.I), "aten::rsqrt"),

    # --- Scatter / masked ops
    (re.compile(r"masked_scatter|launch_masked_scatter_kernel", re.I), "aten::masked_scatter"),
    
]

def map_kernel(name: str):
    for rx, lab in RULES:
        if rx.search(name):
            return lab
    return None

def iter_rows(path: str):
    """
    Robustly parse nsys --format csv output that may include NOTICE/Processing lines.
    Uses csv.reader so quoted kernel names with commas are preserved.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        # filter out NOTICE/Processing lines before feeding into csv
        raw_lines = [ln for ln in f if not ln.startswith("NOTICE:") and not ln.startswith("Processing") and ln.strip()]

    # Find the header line index (contains Total Time, Instances, Name)
    header_i = None
    for i, ln in enumerate(raw_lines):
        if "Total Time" in ln and "Instances" in ln and "Name" in ln:
            header_i = i
            break
    if header_i is None:
        raise RuntimeError("Header not found. Expected columns like: Total Time (ns), Instances, Name")

    reader = csv.reader(raw_lines[header_i:])
    header = next(reader)

    # column indices
    def idx_contains(substr):
        for j, h in enumerate(header):
            if substr.lower() in h.lower():
                return j
        return None

    j_total = idx_contains("Total Time")
    j_inst  = idx_contains("Instances")
    j_name  = idx_contains("Name")

    if j_total is None or j_inst is None or j_name is None:
        raise RuntimeError(f"Missing required columns in header: {header}")

    for row in reader:
        if not row:
            continue
        # guard: some rows could be shorter
        if len(row) <= max(j_total, j_inst, j_name):
            continue
        total_ns = int(row[j_total].replace(",", "").strip())
        calls = int(row[j_inst].replace(",", "").strip())
        name = row[j_name].strip()
        yield name, calls, total_ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_ops_csv", required=True)
    ap.add_argument("--out_unmapped_csv", required=True)
    ap.add_argument("--min_unmapped_ms", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_ops_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_unmapped_csv) or ".", exist_ok=True)

    agg_calls = defaultdict(int)
    agg_ns = defaultdict(int)
    unmapped = []

    for name, calls, ns in iter_rows(args.in_csv):
        lab = map_kernel(name)
        if lab is None:
            unmapped.append((name, calls, ns))
        else:
            agg_calls[lab] += calls
            agg_ns[lab] += ns

    with open(args.out_ops_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aten_op_family", "kernel_calls", "total_cuda_time_ns", "total_cuda_time_ms"])
        for lab in sorted(agg_ns.keys(), key=lambda k: agg_ns[k], reverse=True):
            ns = agg_ns[lab]
            w.writerow([lab, agg_calls[lab], ns, ns / 1e6])

    unmapped.sort(key=lambda x: x[2], reverse=True)
    with open(args.out_unmapped_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_name", "calls", "total_cuda_time_ns", "total_cuda_time_ms"])
        for name, calls, ns in unmapped:
            ms = ns / 1e6
            if ms < args.min_unmapped_ms:
                continue
            w.writerow([name, calls, ns, ms])

    print("Wrote:", args.out_ops_csv)
    print("Wrote:", args.out_unmapped_csv)

if __name__ == "__main__":
    main()