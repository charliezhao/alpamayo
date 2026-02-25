import inspect, torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

m = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, device_map="cpu").eval()

def nparams(mod):
    return sum(p.numel() for p in mod.parameters())

def safe_sig(mod):
    try:
        return str(inspect.signature(mod.forward))
    except Exception as e:
        return f"<could not inspect: {e}>"

print("Top-level children:")
tops = list(m.named_children())
for name, mod in tops:
    print(f"\n== {name} ==")
    print("type:", type(mod))
    print("params (M):", nparams(mod)/1e6)
    kids = [n for n,_ in mod.named_children()]
    print("children:", kids[:30], ("...(truncated)" if len(kids)>30 else ""))
    print("forward:", safe_sig(mod))

total = sum(p.numel() for p in m.parameters())
print(f"\nTOTAL params: {total/1e9:.3f}B ({total/1e6:.1f}M)")

# now break down vlm module:
vlm = m.vlm

visual = vlm.model.visual
lm = vlm.model.language_model
lm_head = vlm.lm_head

items = [
    ("vlm.model.visual", visual),
    ("vlm.model.language_model", lm),
    ("vlm.lm_head", lm_head),
    ("vlm (total)", vlm),
]

print("\nVLM parameter breakdown:")
print(f"{'module':28s} {'params(M)':>12s} {'params(B)':>10s}")
print("-"*54)
for name, mod in items:
    p = nparams(mod)
    print(f"{name:28s} {p/1e6:12.3f} {p/1e9:10.3f}")

# Sanity: show that (visual + lm + lm_head) is close to vlm total
p_sum = nparams(visual) + nparams(lm) + nparams(lm_head)
print("-"*54)
print(f"{'visual+lm+lm_head':28s} {p_sum/1e6:12.3f} {p_sum/1e9:10.3f}")

# Optional: show the delta (extra small modules inside vlm.model, if any)
delta = nparams(vlm) - p_sum
print(f"{'delta vs vlm total':28s} {delta/1e6:12.3f} {delta/1e9:10.3f}")


