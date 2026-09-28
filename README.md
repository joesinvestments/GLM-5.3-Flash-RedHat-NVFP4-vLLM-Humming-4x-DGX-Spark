# GLM-5.3-Flash on 4x DGX Spark: RedHat NVFP4, vLLM main, Humming

GLM-5.3-Flash served across four NVIDIA DGX Sparks (GB10, tensor parallel 4 over RoCE) with:

- **Weights:** [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) at revision `18d55bfd5a2194887738da73753975c9d3842f46` (MIT): NVFP4 experts, BF16 attention.
- **Engine:** vLLM main, nightly `7f1a5398e9` (0.30.1rc1.dev48), not a release, plus our 19-patch series for GB10 and this model, including upstream fixes we carry ahead of vLLM.
- **MoE kernels:** [Humming](https://github.com/vllm-project/humming) (`--moe-backend humming`).
- **Speculative decoding:** the [incoai DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) in 16-bit, standard verification, up to 7 draft tokens.
- **Context:** 500,000 tokens, fp8 KV cache (35 GiB per rank).

Everything here is the exact build we serve: `image-main/build.sh` refuses to build unless every patched file matches the checksums of our production image, and `launch/launch_node.sh` carries our production serving flags.

## Results (2026-09-28)

All numbers are from the production cluster, measured on the day.

| What | How it was measured | Result |
|---|---|---|
| Throughput, 8 concurrent agent sessions | `tools/agent_sessions_long.py`, `LEVELS_ONLY=8`, 3 sweeps: 3-turn tool-calling tasks, ~19K-token prompts, thinking at max, prefix-cache hit rate 0.83 to 0.85 | **52.8 tok/s** aggregate (sweeps 54.6 / 50.0 / 53.9), mean time to first token 3.8 to 4.7 s, draft acceptance 0.32 to 0.37 |
| Single request, long reply | one request writing a 16,000-token reply, thinking at max, temperature 1.0 | 26 to 29 tok/s (559 to 607 s per reply); speculation off: about 23 tok/s (670 to 702 s) |
| Long-reply quality | `tools/long_reply_check`: decode-time logprob of every token against a prefill rescoring of the same token ids, per 2,000-token window | 8 of 8 replies of 16,000 tokens clean at this build's settings: worst window mean \|diff\| 0.091, at most 5 tokens per 1,000 off by more than 1 nat (speculation off measures 0.06 to 0.10) |
| Long prompt | one 94,933-token prompt through the sparse indexer's prefill path, greedy | 38.1 s, correct answer, identical output with and without patch u55222 |
| GPU memory per rank | `nvidia-smi` process memory, fresh boot, idle, median of 3 samples | 95,237 MiB (95,277 on the head node), 1,900 to 1,920 MiB less than without patch u55222 |
| Real agent work | three agent jobs on the served model (a code review, writing and passing unit tests, a decision write-up), each with a scripted pass/fail check | 3 of 3 passed, no server errors, head node never below 8,905 MiB available memory |

## Key findings

### 1. An 8-bit drafter silently corrupted long replies

Our earlier build packed the DFlash2 drafter's weights to 8 bits (patch 10, from a public 8-bit drafter experiment) for speed. Short evals looked fine. Long replies did not: text stayed coherent for a few thousand tokens, then words fused ("resolvesLet", "CycleFollowing"), the reasoning section ended mid-sentence, and the output collapsed. It happened in every configuration we tried with the 8-bit drafter on: block and standard verification, top_p 0.95, async scheduling off, eager mode, one request at a time, prefix caching off, Marlin instead of Humming, the drafter in separate KV memory, and a draft length of 3.

We isolated it one variable at a time:

| Configuration | Long replies |
|---|---|
| DFlash2, 8-bit drafter (any of the settings above) | broken: window mean \|diff\| up to 0.99, up to 269 tokens per 1,000 off by more than 1 nat |
| DFlash2, 8-bit drafter loaded but draft length 0 (nothing verified) | clean |
| The model's own MTP head, draft length 3 | clean |
| DFlash2, 16-bit drafter, draft length 3 | clean |
| DFlash2, 16-bit drafter, draft lengths 7 / 5 / 4 by batch size (this build) | clean |

At a break, single verification steps gave a token the model itself rates as nearly impossible a real probability (for example "Let": decode-time logprob -0.39, prefill -18.86), and it got committed. An instrumented run found 0 mismatches in 5,200 verify steps between the drafted tokens, the tokens the target was fed and the tokens the sampler compared, so the inputs are right. The damage appears only when 8-bit drafts are verified, which points at how the target handles rejected drafts; the root cause is still open. Scope: every run used this build's settings, including `VLLM_MARLIN_USE_ATOMIC_ADD=1`, which changes how the 8-bit drafter's Marlin kernels reduce; we did not test the 8-bit drafter with it off, so this finding is about 8-bit drafting on this stack, not 8-bit drafters in general. Patch 10 remains in the series for anyone who wants to investigate, off by default; do not enable it for serving.

`tools/long_reply_check` is the check that caught this. We now run it every night.

### 2. The best drafter depends on the load

At 1 request, the model's native MTP head (draft length 3) wrote a 16,000-token reply at about 37 tok/s against about 28 for 16-bit DFlash2. At 8 agent sessions, DFlash2 won: MTP ran at 0.925x DFlash2's throughput (sweeps 47.7 / 51.8 / 47.1 against 54.6 / 50.0 / 53.9) and 1.032x its time to first token. MTP's drafts are accepted more often (0.49 to 0.55 against 0.32 to 0.37), but it proposes fewer tokens per step. Measure at the concurrency you actually run.

### 3. 1.9 GB per node back from one upstream fix

The sparse indexer's prefill workspace was sized in tokens, although its cache holds one row per 4 tokens. vLLM reserves it at startup and locks it, so the oversize was held for the life of the server: 1,900 to 1,920 MiB per rank. Patch u55222 carries the workspace half of [vllm#55222](https://github.com/vllm-project/vllm/pull/55222) (open at the time of writing); the other half, an fp8 plan dtype, was already covered by patch 07. On GB10's unified memory that is headroom for the whole node: our head node's available memory after boot went from 9,289 to 10,702 MiB.

### 4. A warm prefix cache inflates throughput numbers

On the same build within the same hour, the same 8-session load read 50.1 tok/s at a 0.84 prefix-cache hit rate and 77.6 tok/s at 0.96, only because an earlier run had left the prompts cached. Compare throughput only between runs with similar hit rates; `agent_sessions_long.py` reports the hit rate for every level.

## Open questions

- **Does patch u55222 cost throughput?** Before it, 6 comparable 8-session sweeps averaged 52.8 tok/s (50.0 to 56.6). After it, one run of 3 sweeps averaged 50.1 (45.6 / 46.7 / 59.1), 0.949x, at the same cache hit rate. The spread is far wider than the gap, the 94,933-token prompt took exactly 38.1 s with and without it, and it only changes a buffer size, so we expect noise, but we have not run a clean side-by-side (fresh boot of each, 3 sweeps each). If you run one, please share it.
- **Why does the 8-bit drafter corrupt the target's output?** See finding 1.
- **Upstream work we did not carry yet:** [vllm#58684](https://github.com/vllm-project/vllm/pull/58684) (removes a GPU-to-CPU sync in sparse-attention planning) waits on [#58260](https://github.com/vllm-project/vllm/pull/58260), an open fix that rewrites the same function; [vllm#58762](https://github.com/vllm-project/vllm/pull/58762) (metadata reuse across KV cache groups) is being replaced by the open draft [#58851](https://github.com/vllm-project/vllm/pull/58851).

## What's here

| Path | What it is |
|---|---|
| `image-main/` | `build.sh`, the patch series (`patches/vllm/series`, 19 patches), two FlashInfer GB10 fixes, `EXPECTED.sha256`, and an optional NCCL build with a RoCE deadlock fix (`nccl/`) |
| `launch/launch_node.sh` | our production serving flags for one rank; only machine-specific values are variables |
| `template/` | Z.ai's official GLM-5.3-Flash chat template plus one explicit thinking opt-out line |
| `docs/patches.md` | every patch: what it does and where it came from |
| `tools/agent_sessions_long.py` | the 8-session agent load behind the throughput numbers (needs a vLLM v0.30.0 checkout for its text pool) |
| `tools/agent_sessions.py` | a shorter agent-session load at 1, 2, 4 and 8 sessions |
| `tools/long_reply_check/` | the long-reply corruption check |
| `tools/ab_harness.py`, `tools/analyze_trace.py`, `tools/quality_eval.py` | drafter A/B harness, trace analysis, NLL and GSM8K quality eval |
| `tools/lab/` | kernel and NCCL microbenchmarks and an oomwrap supervisor |

## Quick start

**1. Build the image on each node.** You need the base image locally (about 20 GB):
```bash
docker pull vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423
cd image-main && ./build.sh      # builds glm53-flash-gb10:main-7f1a
```
The script copies the target files out of the base image, applies the series in order plus the FlashInfer patches with zero fuzz, and refuses to build unless every result matches `EXPECTED.sha256`. The build only copies files; no container runs. To include the NCCL fix, run `nccl/build_nccl.sh` first (about 12 minutes on a Spark) and then `NCCL_LIB=nccl/libnccl.so.2 ./build.sh`.

**2. Get the weights onto every node** (not redistributed here; follow each license):
- Checkpoint: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) at revision `18d55bfd5a2194887738da73753975c9d3842f46`, in `$HF_ROOT/hub/redhat-glm53-flash-nvfp4/`.
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at revision `7d74cdd881ed7e32c31175984a67823127b66cfe`, in `$HF_ROOT/hub/glm53-flash-dflash2/`. We chose it over the two later revisions (`dc77ff1`, `bf582e4`) in A/Bs on an earlier stack (a different checkpoint, the 8-bit drafter); we have not re-run that comparison on this checkpoint.

Download once and copy node to node over the fast link rather than downloading four times.

**3. Put the chat template next to the checkpoint** on every node:
```bash
cp template/chat_template_zai0907_optout.jinja "$HF_ROOT/hub/redhat-glm53-flash-nvfp4/"
```
The launcher passes it with `--chat-template` and refuses to start without it. It is Z.ai's official 09-07 template plus one explicit opt-out line: every request reasons unless it sends `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, it renders byte for byte like the official template.

**4. Launch, workers first, head last:**
```bash
export NODES="<ip0> <ip1> <ip2> <ip3>"   # rail IPs, rank 0 (head) first
./launch/launch_node.sh 3   # on node 3, then 2 and 1
./launch/launch_node.sh 0   # on the head node
```
Defaults assume the Spark's QSFP port shows up as `enp1s0f0np0` / `enP2p1s0f0np0` (RDMA devices `rocep1s0f0` / `roceP2p1s0f0`) and RoCE GID index 3; see the variables at the top of the script. Point `CACHE` at a fresh directory for each image build (it holds the JIT caches). We run every rank under [oomwrap](https://github.com/osolmaz/oomwrap) with `tools/lab/oomwrap_supervise.sh`.

**5. Verify with a real completion,** not just `/health`:
```bash
curl -s http://HEAD:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":16,"messages":[{"role":"user","content":"Say OK"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```
For quality, run `tools/long_reply_check` against your server before trusting long outputs.

**Memory:** each Spark runs earlyoom (SIGTERM at 4% free, about 4.9 GB). The 35 GiB KV cache keeps the head node clear of that line.

## Credits

- **[Red Hat AI](https://huggingface.co/RedHatAI)**: the GLM-5.3-Flash NVFP4 checkpoint served here (MIT). **[Z.ai](https://huggingface.co/zai-org) ([@Zai_org](https://x.com/Zai_org))**: GLM-5.3-Flash (MIT) and the official chat template.
- **[Humming](https://github.com/vllm-project/humming)** ([Jinzhen Lin](https://github.com/jinzhen-lin), [Julian Huang](https://github.com/huangzhilin-hzl), [Misha Goin](https://github.com/mgoin) ([@mgoin_](https://x.com/mgoin_)) and the Humming contributors, Apache-2.0): the MoE kernels.
- **[vLLM](https://github.com/vllm-project/vllm) ([@vllm_project](https://x.com/vllm_project))** and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** (Apache-2.0). vLLM contributors whose pull requests we carry: [Matt Mastracci](https://github.com/mmastrac) ([@mmastrac](https://x.com/mmastrac); #58704, #58454), [Juntian777](https://github.com/Juntian777) (#57632), [shiweijiezero](https://github.com/shiweijiezero) (#58834), [QHarshil](https://github.com/QHarshil) (#58021), [drakosha](https://github.com/drakosha) (#55222), [JaredforReal](https://github.com/JaredforReal) (#57477, now in vLLM main), and vLLM's own revert #58250 of [mgoin](https://github.com/mgoin)'s fused DFlash2 grouped convolution (#55960).
- **Tony ([@2WildTech](https://x.com/2WildTech), [tonyd2wild](https://github.com/tonyd2wild))**: the GB10 patch set that patches 01 to 08 are ported from, and the two FlashInfer FP8 MLA fixes.
- **[incoai](https://huggingface.co/incoai)**: the GLM-5.3-Flash DFlash2 drafter (CC BY-NC-ND 4.0), built on DFlash from [Z Lab](https://github.com/z-lab/dflash) ([@zhijianliu_](https://x.com/zhijianliu_)).
- **[Jacopo Nardiello](https://github.com/jnardiello) ([@jnardiello](https://x.com/jnardiello))**: the 8-bit drafter experiment that patch 10 ports and the 8-bit residue idea behind patch p2 (both off in this build).
- **NCCL:** [Stanislav Bardyuk](https://github.com/kodlan) wrote [NVIDIA/nccl#2393](https://github.com/NVIDIA/nccl/pull/2393), the fix for the RoCE deadlock reported in [#2334](https://github.com/NVIDIA/nccl/issues/2334); [Rami Nudelman](https://github.com/raminudelman) at NVIDIA brought it to us.
- **[oomwrap](https://github.com/osolmaz/oomwrap)** by [Onur Solmaz](https://github.com/osolmaz) ([@onusoz](https://x.com/onusoz)): memory-pressure supervision for every launch.

## License

The scripts, tools and patches in this repo are Apache-2.0 (see `LICENSE`); the patches modify Apache-2.0 vLLM and FlashInfer code and BSD-3-Clause NCCL code. Humming (Apache-2.0) is not included here; the build selects it with `--moe-backend humming`. `template/` is Z.ai's official chat template with one added line and keeps Z.ai's license. No model weights are included. Each model has its own license: RedHat's checkpoint is MIT, and the incoai drafter is non-commercial and no-derivatives.
