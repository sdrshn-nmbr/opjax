# Kevin: Multi-Turn RL for Generating CUDA Kernels

Carlo Baronio∗ †
Stanford University
Cognition AI
cbaronio@stanford.edu
carlo@cognition.ai
Pietro Marsella∗ †
Stanford University
Cognition AI
marsella@stanford.edu
pietro@cognition.ai
Ben Pan∗ †
Stanford University
Cognition AI
benpan@stanford.edu
ben@cognition.ai
Simon Guo
Stanford University
simonguo@stanford.edu
Silas Alberti
Cognition AI
silas@cognition.ai
## Abstract

Writing GPU kernels is a challenging task and critical for AI systems’ efficiency.
It is also highly iterative: domain experts write code and improve performance
through execution feedback. Moreover, it presents verifiable rewards like cor-
rectness and speedup, making it a natural environment to apply Reinforcement
Learning (RL). To explicitly incorporate the iterative nature of this process into
training, we develop a flexible multi-turn RL recipe that addresses unique chal-
lenges encountered in real-world settings, such as learning from long trajectories
and effective reward attribution across turns. We present Kevin - K(ernel D)evin,
the first model trained with multi-turn RL for CUDA kernel generation and op-
timization. In our evaluation setup, Kevin shows significant gains over its base
model (QwQ-32B), improving correctness of generated kernels (in pure CUDA)
from 56% to 82% and mean speedup from 0.53x to 1.10x of baseline (PyTorch
Eager), and surpassing frontier models like o4-mini (0.78x). Finally, we study its
behavior across test-time scaling axes: we found scaling serial refinement more
beneficial than parallel sampling. In particular, when given more refinement turns,
Kevin shows a higher rate of improvement.
## 1 Introduction

Writing efficient GPU kernels [8, 59, 56] in domain-specific languages (CUDA [29], Triton [48],
ThunderKittens [42], CUTLASS [32], etc.) is critical for enabling AI systems’ efficiency at scale,
yet it remains difficult and costly due to the deep domain expertise required. This has led to a surge
of interest in exploring how Large Language Models (LLMs) could help generate GPU kernels
[33, 22, 31] using agentic systems [6, 5, 28, 21, 15] that leverage extensive test-time compute. These
inference-based approaches are inherently limited by the base models’ capability in this domain. On
the other hand, the presence of verifiable rewards in the form of correctness and speedup against a
reference implementation makes reinforcement learning (RL) a natural approach. This leads to our
investigation: How can we train a model using RL to solve the real-world engineering task of CUDA
kernel generation?
GPU kernel generation emphasizes not just functional correctness, but more importantly performance
— distinguishing this code optimization problem from binary-reward tasks that involve passing unit
tests [17] or producing an acceptable proof [60]. Since speedup is a continuous goal, performance
∗
Equal contribution.
†
Work done during an internship at Cognition AI.
Preprint version for arXiv.
arXiv:2507.11948v1 [cs.LG] 16 Jul 2025

**==> picture [Figure 1] intentionally omitted <==**

Figure 1: Within each training step, the model iteratively generates, executes, and refines kernels
over multiple turns. Kernels are rewarded individually, based both on their performance and their
contribution to subsequent speedups: K1, for example, while incorrect, leads to both a correct, slow
kernel, K2, and a correct, performant kernel, K3, and should thus be rewarded accordingly. This setup
enables Kevin to learn advanced code generation strategies that span multiple turns. Note: CoT’ is
the summarized chain of thought (CoT).
engineers take an iterative approach: they conduct many rounds of optimization based on previous
kernel code, its execution result, and timing profiles. Hence, arriving at an optimized solution relies
on multiple turns conditioned on previous execution feedback. In contrast, popular RL methods to
train LLMs on verifiable rewards [38, 20] rely on the outcome reward of a single turn (“single-turn
RL training”). We hypothesize that explicitly incorporating successive turns of code generation,
execution, and feedback into each RL training step (“multi-turn RL training”) better mirrors the
iterative nature of kernel development, helping the model to learn more advanced code generation
strategies that span multiple refinement turns.
We design a simple yet effective multi-turn RL training recipe, shown in Figure 1, that addresses the
key challenges of training for CUDA kernel generation and optimization:
1. Long trajectories lead to sparse rewards and context explosion. To improve sample
efficiency, we split trajectories and use each turn as an individual training sample. To address
context explosion from long CoTs while preserving reasoning information, we summarize
CoTs of prior turns.
2. Finding an optimal solution may require rewarding suboptimal kernels that eventually
lead to more performant ones. Therefore, we study approaches to aggregate intermediate
rewards across turns, finding a configuration that balances the correctness-performance
trade-off.
3. Reward hacking is prevalent as kernel generation is an open-ended, real-world engi-
neering task: e.g. the model can trick the evaluation harness, lazily copying the reference
implementation instead of actually implementing kernels. To prevent this, we analyze the
model’s failure modes and enforce strict rule-based checks.
Enabled by our multi-turn RL training method on 180 KernelBench tasks from Level 1 and 2, we
present K(ernel D)evin, the first RL-trained model to generate CUDA kernels. We compare Kevin and
other models in our evaluation setting on a representative KernelBench eval set. Kevin improves upon
its base model (QwQ-32B [47]) both in correctness (56% → 82%) and mean speedup of generated
kernels (in pure CUDA): from 0.53x to 1.10x over PyTorch Eager, while outperforming frontier
models like OpenAI o4-mini (0.78x).
We then study the characteristics of Kevin in a test-time scaling setting, comparing it to a single-turn
RL-baseline. We systematically compare the benefits of scaling along two axes of test-time compute:
sequentially with more refinement turns [12, 51] or in parallel with more trajectories [2, 41]. In our
setting, we find that sequential scaling is much more effective, highlighting the importance of iterating
upon execution feedback. We observe that the model trained with multi-turn RL exhibits better
scaling characteristics with more refinement turns, compared to the base model and the single-turn
RL baseline. Our core contributions include:
1. We design an effective yet flexible multi-turn RL training strategy that significantly
improves model’s capability on CUDA kernel generation. This strategy addresses
challenges that arise in real-world settings, and may be applicable to other environments
that benefit from iterative optimizations.
2. We found multi-turn is more effective both for training and inference through systematic
ablations: the multi-turn trained model outperforms the single-turn trained model across
different evaluation setups. Furthermore, we found multi-turn inference is more effective
across both models under a fixed inference budget.
3. Kevin exhibits strong test-time scaling trends on both serial and parallel axes, with a
faster rate of improvement than its single-turn RL counterpart and its base model, while
maintaining exploration capacity.
## 2 Background and Related Work

### 2.1 LLM for GPU Kernel Optimization

There has been a surge of interest in exploring how to leverage LLMs to generate GPU kernels [31],
driven by the high cost and the long engineering cycles required to develop them (e.g. 2 years for
efficient FlashAttention [7] port after Hopper GPU release). However, frontier models underperform
on representative benchmarks like KernelBench [33] and TritonBench [22], likely due to GPU code
being severely underrepresented in the training data (CUDA, for example, accounts for less than
0.01% of pretraining data in the Stack [18, 23]). Collecting more expert-written code is expensive, as
only a limited number of developers are able to implement high-quality kernels. To tackle this task,
there has been an explosion of agentic systems [6, 5, 28] with custom workflows and evolutionary
search methods [21, 15]. Yet these approaches typically incur high inference cost — e.g. $15 per
kernel [21]. Improving the base LLM’s kernel-generation ability is therefore essential — and could
significantly boost the efficiency for downstream agentic workflows.
### 2.2 RL Optimization for LLMs Targeting Verifiable Domains

Reinforcement Learning techniques like GRPO [38] have been shown to significantly enhance
LLMs’ performance on verifiable domains [20] such as math [45, 52] and competitive programming
[47, 26, 27]. These approaches can be further adapted for real-world software tasks, using fine-grain
unit tests [24] or comparisons between code edits [54] as outcome rewards. Existing methods for code
optimizations — where objective concerns performance beyond correctness — have been largely
confined to supervised fine-tuning [49] and imitation learning [39], highlighting Kevin’s RL approach
a novel contribution for this setting.
Given that tasks like performance optimization or long-horizon planning require multiple sequences
of interrelated actions, several works [14, 3, 53, 61, 62] have explored RL training for multi-turn
optimizations beyond optimizing for outcome from a single turn. Specific for the code setting,
RLEF [13] frames code generation as a multi-turn RL task: the model is allowed a fixed number of
refinements turns and assigned a single binary pass/fail reward for final generation — training with
such an approach might present sample-inefficiency issues. Unlike RLEF, which assigns rewards
only at the final turn, our multi-turn RL framework for Kevin trains on every turn regardless of how
optimal the code is, and optimizes for performance beyond just correctness. It is worth noting that
Kevin’s multi-turn RL training could be viewed as a variant of Meta-Learning [55, 11] or In-Context
Reinforcement Learning [30, 43, 36], where the focus is to improve solution quality during test-time
with feedback [34]; but adapted in a novel way to the challenging real-world setting of GPU kernel
generation and code optimization.
## 3 Task and Baseline

### 3.1 Environment and Evaluation

We use KernelBench [33], a popular benchmark for evaluating LLMs’ ability to generate CUDA
kernels for deep learning workloads in PyTorch. We chose 180 of the 100 Level 1 problems (basic
operators: convolutions, matrix multiplies, loss functions, etc.) and 100 Level 2 problems (sequences
of operators with fusion opportunities) as training environments. Since current KernelBench does
not provide a train-test split, we construct 80 additional novel tasks following the same methodology
(see Appendix A). We build the evaluation set by combining our 80 newly created tasks with the 20
remaining original KernelBench tasks, for a total of 100 evaluation tasks.
Each KernelBench task consists in generating a CUDA kernel given a PyTorch reference imple-
mentation, which is used to evaluate correctness and speedup. In our set up, we evaluate the
model-generated kernels as follows: we verify the output is in the correct format (ensure resultant
code is only implemented with inline CUDA) and check for reward hacking (Section 6.2). We then
evaluate the kernel for compilation, runtime errors, and correctness. If the implementation is correct,
we profile the kernel for its runtime.
### 3.2 Kernel Score Design

Asweareconcernedbothwithcorrectnessandspeedup, weassignascoreS foreachkernelevaluation
result that effectively balances the correctness-performance tradeoff.
S = 0.3 · 1{correct} +
Tbaseline
Tkernel
· 1{correct}
Correctness is checked against the reference program when tested with randomized inputs; speedup
is computed as the ratio between PyTorch baseline time and kernel runtime. We experimented with
various weights of correctness and speedup, finding this configuration through ablations on models
ranging from 7B to 32B.
In addition, we explored rewarding intermediate objectives (successfully compile or execute), yet this
caused model to over-optimize for intermediate steps (e.g. generating kernels that only compile, but
aren’t necessarily correct). We also experimented with a length penalty on the response, as suggested
by [44], but found that it degrades our model’s performance during training.
### 3.3 Single-Turn Training

We apply GRPO [38] to train the model on kernel generation without iterating on external feedback
("single-turn" training). In each training step, we sample 16 responses per task and assign the
evaluated score as the reward for each kernel. We compute the GRPO loss according to [38], which
updates the policy by maximizing the following objective:
JGRPO(θ) = E[q ∼ P(Q),{oi}G
i=1 ∼ πθold
(O|q)]
1
G
G X
i=1
1
|oi|
|oi|
X
t=1
(
min
"
πθ(oi,t|q,oi,<t)
πθold (oi,t|q,oi,<t)
Âi,t,clip

πθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)
,1 − ϵ,1 + ϵ

Âi,t
#
− βDKL(πθ||πref)
)
(1)
where Âi,t = ri−mean(r)
std(r) , and ri is the score of a specific kernel.
We note the importance of using a base model with strong enough priors to obtain a non-sparse
reward for correctness and speedup in the beginning of training. For instance, training on
DeepSeek-R1-Distill-Qwen7B [9] exhibited reward hacking (see Section 6.2) and failed to learn.
Hence, we use a stronger base model, Qwen QwQ-32B [47]. We perform two gradient steps for a
batch (1 on-policy, 1 off-policy) following [38]. We use max_response_length = 16384.

**==> picture [Figure 2] intentionally omitted <==**

Figure 2: Reward plateaus during
single-turn training. We trained up to
step 50 (100 gradient steps).
Following [57], we apply Clip-Higher, decoupling the
lowerandhigherclippingrange(0.2and0.28respectively).
We sample with temperature = 0.9 for both training
and inference. We set the KL coefficient to 0 to allow the
model to deviate freely from the base policy, following
[26].
We observe that reward plateaus after 50 steps, likely be-
cause single-turn training prevents the model from refining
its kernels. Many generated kernels are nearly correct–
often a syntax or compilation fix away–but still receive
0 reward, discouraging the model from producing them.
Similarly, the correct kernels do not achieve high speedup,
as the model optimizes for correctness rather than attempt-
ing a risky approach. We address these limitations through
multi-turn training.
## 4 Multi-Turn Training

In each multi-turn training step:
1. For each task, we sample m parallel trajectories with n refinement turns. To improve sample
efficiency, each refinement turn (CoT + response) in a trajectory becomes a single training
sample. The response of the model after the CoT consists of a kernel and a CoT summary.
2. We construct the context of a sample by including the history of previous responses, which
include generated kernels along with their summarized CoTs, and evaluation feedback.
3. We evaluate the generated kernel and compute its score as shown in Section 3.2. The reward
of each turn (CoT + response) is the discounted sum of current and subsequent scores, which
we elaborate in Section 4.3.
4. For each task, we normalize the rewards across the mn samples for advantage calculation.
Then we compute the GRPO loss over the entire batch.
### 4.1 Managing Context

Reasoning models generate long CoTs, especially for complex tasks like kernel generation. Including
all CoTs causes the context to grow rapidly, reaching 50-100k tokens within a few turns, surpassing
the model’s context length. To prevent context explosion, we discard CoTs of previous turns;
yet to preserve information regarding the reasoning process, we ask the model to summarize the
changes applied. This summary, along with the generated kernels and evaluation results, is passed to
subsequent turns.
### 4.2 Training On Every Refinement Turn

In a naive implementation, each n-turn trajectory is a single training sample. To improve sample
efficiency, we split a n-turn trajectory into n training samples, each corresponding to the kernel +
CoT summary of a refinement turn with the context containing the history. Hence, the kernel and
CoT summary receives the reward of that particular turn.
### 4.3 Reward Aggregation and Discounting

We initially explored two naive strategies for multi-turn credit assignment. The greedy approach
assigns to each turn its corresponding kernel score, while the outcome-based approach assigns to all
turns the best score in the trajectory. The former failed to reward early suboptimal turns that lead to
performant kernels later, while the latter ignores individual contributions and is sample inefficient.
Our method balances both approaches by aggregating the future kernels scores with a discount factor.
We conduct ablations on the reward formulation. For score aggregation, we can either take the
sum Rt =
PT
i=t γi−t
ri or maximum Rt = maxi=t,...,T

γi−t
ri over future scores. Sum favors
generating multiple good kernels, while max prioritizes achieving one high-performing kernel. We
evaluate both forms with γ = 0.4 and γ = 0.8.
Experiments show that sum with γ = 0.4 scales best over 8 turns, though max performs better with
γ = 0.8 with fewer turns.

**==> picture [Figure 3] intentionally omitted <==**

Figure 3: Sum with γ = 0.4 is the most effective reward formulation. Here we evaluate models
trained with different reward formulations (Sum vs Max aggregation across turns and discount factor
γ = 0.4 vs γ = 0.8) with 16 parallel trajectories and 8 refinement turns. We compare how each setup
scale with refinement turns. Though Sum with γ = 0.4 achieves lower performance and correctness
in the first turn, it exhibits the best scaling behavior overall.
### 4.4 Multi-Turn Training Behavior

**==> picture [Figure 4] intentionally omitted <==**

Figure 4: Reward climbs steadily for
multi-turn training. We train up to 40
steps (80 gradient steps).
For our final training run for Kevin, we use 16 parallel
trajectories and 4 refinement turns per task. Each batch
contains 8 tasks. We use the sum reward formulation with
discount factor γ = 0.4.
Unlike single-turn training, reward now steadily increases.
We also observe response length behaviors similar to [27]:
the response length initially decreases, and then it starts
increasing again as the model attempts more sophisticated
solutions. Following [27], we extend the max response
length from 16K to 22K tokens at step 30.
## 5 Evaluation

As kernel generation is a challenging task, models are often given extensive test-time compute to
tackle it. In our inference setting, we employ multiple parallel trajectories, where each trajectory is
made up of several serial turns.
We mark a given trajectory correct if it contains at least one correct kernel. Its performance is
the speedup of the fastest kernel (within the trajectory) over the PyTorch Eager reference (speedup
of 0x if no kernel is correct). We also consider the fastp metric, introduced by [33], which is a
binary indicator for whether a trajectory contains a correct kernel with performance of p or more. To
aggregate a metric across k parallel trajectories for a given task, we compute: best@k, the maximum
for that metric across all trajectories; avg@k, the average value across trajectories.
### 5.1 Result on KernelBench Eval Set

We compare Kevin against frontier models and the single-turn RL baseline on our aforementioned
KernelBench eval set of 100 tasks (Section 3.1), with 16 parallel trajectories, 8 serial refinement turns.
As shown in Table 1, Kevin achieves a higher performance than its single-turn trained counterpart
and other frontier models, demonstrating significant improvement from its base model (QwQ-32B).
Qualitatively, Kevin is able to more effectively implement more aggressive optimizations across
several turns (see Appendix G for examples and more details).
Model Correctness Performance fast1 fast1.5
best@16 avg@16 best@16 avg@16 best@16 avg@16 best@16 avg@16
Kevin (Multi-Turn) 82% 46% 1.10x 0.40x 43% 15% 20% 6%
Single-Turn RL 82% 45% 0.85x 0.35x 43% 16% 16% 4%
Qwen QwQ-32B 56% 11% 0.53x 0.08x 23% 3% 10% 1%
OpenAI o4-mini 38% 22% 0.78x 0.27x 21% 7% 13% 6%
OpenAI o3-mini 27% 8% 0.30x 0.08x 9% 2% 4% 2%
Table 1: Kevin, trained with our multi-turn RL setup, outperforms other models in terms of
correctness and performance. Here we evaluate models on 100 unseen KernelBench tasks, under a
test-time compute setup of 16 parallel trajectories with 8 refinement turns each trajectory.
### 5.2 Scaling Refinement Turns

Leveraging execution feedback is crucial at test time [12, 51]. Thus, we evaluate how Kevin scales
with additional refinement turns. As shown in Figure 5, the single-turn model achieves slightly better
performance with 1 turn, as its training objective optimizes for a single attempt. However, when
given more refinement turns, the multi-turn trained model achieves significantly higher performance,
with its curve showing the highest slope. This shows that multi-turn training enhances the model’s
ability to refine and optimize kernels over turns.
### 5.3 Scaling Parallel Samples

We study how best@k performance scales when increasing the number of parallel trajectories k, while
fixing the number of serial refinements turns. Prior work for RLVR on math problems [58] found
that RL training limits models’ exploration capacity, leading to worse best@k metrics than the base
model at large k. As shown in Figure 6, the performance curve of the single-turn RL model presents a
lower slope compared to the base model, possibly hinting at this phenomenon. In contrast, our model
trained with multi-turn RL achieves a higher slope compared to both the single-turn counterpart and
the base model, suggesting that multi-turn training could maintain model’s exploration capacity while
improving model’s performance.

**==> picture [Figure 5] intentionally omitted <==**

Figure 5: Kevin effectively leverages multiple
turns. We evaluate the above checkpoints under
the same environment with 16 parallel trajectories
and 8 refinement turns. We observe that both Kevin
and single-turn RL model significantly improves
upon QwQ-32B, but the performance curve for
Kevin is steeper than the single-turn model.

**==> picture [Figure 6] intentionally omitted <==**

Figure 6: Multi-turn training maintains explo-
ration capacity. Refinement turns are fixed to 8,
and best@k performance is computed with the esti-
mator according to [4]. As k increases, the best@k
performance continues to increase for multi-turn
trainedmodel, butitplateausforsingle-turntrained
model.
### 5.4 Parallel vs Sequential Scaling

As scaling test-time compute through parallel sampling [41] and sequential iterative refinement [12]
are both helpful, we want to systemically compare the two and understand which is more effective in
the context of kernel generation. To investigate, we evaluate 3 inference-time configurations under the
same total inference call budget (128 generated kernels): 128 trajectories with 1 turn, 32 trajectories
with 4 turns, and 16 trajectories with 8 turns. As shown in Table 2, we find that in our experimental
setup, allocating more refinement turns during test-time is consistently better across the multi-turn
RL trained, single-turn RL trained, and base model, with 16 trajectories and 8 turns being the most
optimal configuration for most cases.
As demonstrated in Section 5.1, multi-turn outperforms single-turn training when evaluated in a multi-
turn inference setting. However, given that single-turn training optimizes for single-turn performance,
a natural question arises: does the single-turn trained model perform better by generating more
single-turn responses in parallel? In Table 2, we observe that in a single-turn inference setting with
128 parallel trajectories, the single-turn model achieves slightly better performance than the multi-turn
model. However, when given more refinement turns at inference, the performance and correctness
improve for all models. This strengthens the case for training a model that could use feedback
effectively across multiple turns. Moreover, the multi-turn trained model achieves significantly higher
performance, with faster improvement rates compared to the single-turn trained model. This shows
that multi-turn training enhances the model’s ability to improve performance over turns at test-time.
Inference Config Performance Correctness
Model Total # Traj # Turns pass@128 pass@128
Multi-Turn RL 128 16 8 1.10x 82.00%
Multi-Turn RL 128 32 4 1.02x 83.00%
Multi-Turn RL 128 128 1 0.65x 76.00%
Single-Turn RL 128 16 8 0.85x 82.00%
Single-Turn RL 128 32 4 0.81x 79.00%
Single-Turn RL 128 128 1 0.70x 73.00%
QwQ-32B 128 16 8 0.53x 57.00%
QwQ-32B 128 32 4 0.47x 52.00%
QwQ-32B 128 128 1 0.42x 54.00%
Table 2: Multi-turn inference with 16 trajectories and 8 turns is our most optimal setup. Here we
compare inference configurations and their corresponding performance (× speedup) and correctness
rates, on multi-turn (Kevin), single-turn RL trained models, and base model QwQ-32B.
## 6 Discussion

### 6.1 Diagnosing Model Instability

**==> picture [Figure 7] intentionally omitted <==**

Figure 7: "Not okay ratio" is a proxy for model
instability and predicts junk. It starts rising
around step 48, while junk appears 15 steps later.
"Clipping Ratio" [27] reflects responses truncated
for junk.
We observe that training for longer often leads
to the model producing repetitive and nonsensi-
cal outputs ("junk"). During multi-turn training,
the junk first appears in the final turn and grad-
ually spreads to earlier turns, resulting in model
collapse afterwards.
To investigate this issue, we identified a proxy
signal, which we call the "Not Okay Ratio".
QwQ-32B always begins its chain of thought
with "Okay, " but after 40 steps of training, the
model begins with erratic variants like "Okay
Amigos, so I need to optimize this
3D tensor-matrix multiplication" and
"Okay Holy crap, I need to get this code optimized". These "Not Okay" responses
indicate early signs of model instability and appear well before junk, making the "Not Okay Ratio" a
valuable predictor.
We initially tried to mitigate instability by adding a KL penalty (0.001 and 0.01) to the GRPO loss,
but it slowed learning (see Appendix E) without affecting model stability. Instead, by using constant
length normalization in the GRPO loss [25] and gradient norm clipping of 0.05, we are able to
effectively delay the onset of junk until step 100.
### 6.2 Reward Hacking

In our early experiments with smaller models like DeepSeek-R1-Distill-Qwen-7B, we observed
frequent reward hacking: the model calls the reference implementation (PyTorch) by directly copying
it, wrapping it in try-except statements, or inheriting the reference implementation method.
Reward hacking typically emerges when the model capabilities falls short of task difficulty [1]. In
our setting, when model fails to produce correct kernels, the "hacked’ kernels are likely the only ones
receiving positive reward and get disproportionately reinforced due to advantage normalization. To
prevent this, we upgraded our base model to the more capable QwQ-32B model as a stronger prior.
However, we observe instances of reward hacking even for stronger models. For Level 2 tasks
(targeting kernel fusion), we observe that the model only fuses simple operators (e.g. ReLU, Max),
leaving the operator worth optimizing (e.g. convolutions) unfused and unmodified (left in PyTorch).
To prevent this, we impose stricter format checks that assign 0 reward to responses with any PyTorch
functional operators. We elaborate more on concrete examples of reward hacks in the Appendix F.
### 6.3 Data Distribution

We found it critical to have a balanced difficulty distribution across the dataset, so that
on average each batch contains both easier and harder tasks. In one experiment with
DeepSeek-R1-Distill-Qwen-14B [9], we trained on a subset of only easy tasks. We observed
that the reward quickly plateaus as the model overfits to a single difficulty level. Thus, we address
this issue by using a stronger base model QwQ-32B and training on both level 1 and 2 of the dataset,
which contained tasks with a variety of difficulty and associated optimization techniques.
## 7 Conclusion

### 7.1 Summary

We designed a multi-turn RL training recipe that addresses challenges when applied to the real-world
task of kernel generation: specifically, effective context management and credit attribution across
every turn to enable better sample efficiency. We also implemented mechanisms to prevent reward
hacking, found an interesting proxy reward to diagnose instability, and experimented with approaches
to constrain this issue.
We present Kevin, the first model trained with RL to generate CUDA kernels, on KernelBench Level
1 and 2 tasks. Evaluated on an unseen KernelBench evaluation set, Kevin outperforms its single-turn
RL counterpart and frontier models, demonstrating that our training recipe enables the model to
learn more effective refinement strategies. Multi-turn training also enables better test-time scaling,
both when increasing sequential refinement and parallel sampling compute, while preserving the
exploration capacity of the model.
### 7.2 Limitations

Since the base model (QwQ-32B [47]) is already heavily post-trained, additional RL training could
easily destabilize it [46]. Due to limited compute and long RL training time for this task, we perform
training up to 80 gradient steps. Consequently, we were unable to run more exhaustive ablations (e.g.
varying the number of turns during Multi-Turn RL) and defer those studies to future work.
We further note limitations regarding kernel optimizations. As KernelBench tasks are specified with
a specific pre-defined tensor input size, the speedups we measure in Section 3.2 are only accurate for
those dimensions on our NVIDIA H200 GPUs.
### 7.3 Future Work

We outline several directions for extending our method. Incorporating a learned value network and
using Proximal Policy Optimization [37] might improve the baseline estimation during training. At
training and test-time, we could implement more sophisticated search methods such as beam search
or Monte-Carlo Tree Search [40]. Inspired by recent works [35], we could also leverage the value
network as a verifier for search at test-time.
Our multi-turn RL process demonstrates success in the real-world engineering task of GPU kernel
generation. However, we designed this recipe in a flexible manner, potentially applicable to a wider
range of tasks that feature verifiable rewards and execution feedback across a trajectory. We believe
explicitly training models to reason about complex tasks over multiple turns to be a key step towards
enabling autonomous AI systems.
## 8 Acknowledgment

We are grateful to Anna Goldie, Alex Zhang, Allen Nie, Anikait Singh, Anne Ouyang, Azalia
Mirhoseini, Benjamin Spector, Bradley Brown, Jerry Liu, Jordan Juravsky, Justus Mattern, Karina
Nguyen, Konwoo Kim, Lynn Cherif, Jon Saad-Falcon, Michael Zhang, Neil Band, Qijing Jenny
Huang, Shayan Talaei, William Li, Shafin Khan, Tatsunori Hashimoto for their helpful discussions
and feedback throughout this project.
## References

[1] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.
Concrete problems in ai safety, 2016.
[2] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré,
and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated
sampling, 2024.
[3] Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao
Wang, Akshay Malik, Graham Neubig, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz,
Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Skyrl-v0: Train real-world long-horizon
agents via reinforcement learning, 2025.
[4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto,
Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul
Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke
Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad
Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias
Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex
Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain,
William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra,
Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer,
Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech
Zaremba. Evaluating large language models trained on code, 2021.
[5] Terry Chen, Bing Xu, and Kirthi Devleker. Automating gpu kernel generation with deepseek-
r1 and inference-time scaling. https://developer.nvidia.com/blog/automating-gpu-
kernel-generation-with-deepseek-r1-and-inference-time-scaling/, February
2025. Accessed: 2025-05-15.
[6] Sana Damani, Siva Kumar Sastry Hari, Mark Stephenson, and Christos Kozyrakis. Warpdrive:
An agentic workflow for ninja gpu transformations. In Proceedings of the Machine Learning
for Systems Workshop at NeurIPS 2024, 2024. Accessed: 2025-05-15.
[7] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023.
[8] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast
and memory-efficient exact attention with io-awareness, 2022.
[9] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning, 2025.
[10] Tulsee Doshi. Gemini 2.5: Our most intelligent models are getting even bet-
ter. https://blog.google/technology/google-deepmind/google-gemini-updates-
io-2025/, May 2025. Accessed: 2025-05-21.
[11] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl2
:
Fast reinforcement learning via slow reinforcement learning, 2016.
[12] Ryan Ehrlich, Bradley Brown, Jordan Juravsky, Ronald Clark, Christopher Ré, and Azalia
Mirhoseini. Codemonkeys: Scaling test-time compute for software engineering, 2025.
[13] Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Quentin Carbonneaux, Taco Cohen,
and Gabriel Synnaeve. Rlef: Grounding code llms in execution feedback with reinforcement
learning, 2025.
[14] Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, and Christopher D. Manning. Synthetic
data generation & multi-step rl for reasoning & tool use, 2025.
[15] Google DeepMind. Alphaevolve: A gemini-powered coding agent for designing advanced
algorithms, May 2025. Accessed: 2025-05-15.
[16] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf:
An easy-to-use, scalable and high-performance rlhf framework, 2024.
[17] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and
Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024.
[18] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz
Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau,
Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code,
2022.
[19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large lan-
guage model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium
on Operating Systems Principles, 2023.
[20] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze
Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya
Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris
Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh
Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training, 2025.
[21] Robert Tjarko Lange, Aaditya Prasad, Qi Sun, Maxence Faldor, Yujin Tang, and David Ha.
The ai cuda engineer: Agentic cuda kernel discovery, optimization and composition, 2025.
Accessed: 2025-05-15.
[22] JianlingLi, ShangzhanLi, ZhenyeGao, QiShi, YuxuanLi, ZefanWang, JiachengHuang, Haojie
Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. Tritonbench: Benchmarking
large language model capabilities for generating triton operators, 2025.
[23] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao
Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii,
Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João
Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee,
Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang,
Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan
Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha
Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav
Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank
Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Dan-
ish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz
Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.
Starcoder: may the source be with you!, 2023.
[24] Jiate Liu, Yiqin Zhu, Kaiwen Xiao, Qiang Fu, Xiao Han, Wei Yang, and Deheng Ye. Rltf:
Reinforcement learning from unit test feedback, 2023.
[25] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and
Min Lin. Understanding r1-zero-like training: A critical perspective, 2025.
[26] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaox-
iang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada
Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini
level. https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-
14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51, 2025. Notion
Blog.
[27] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan
Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Sto-
ica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl.
https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-
with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2, 2025.
Notion Blog.
[28] METR. Measuring automated kernel engineering, February 2025. Accessed: 2025-05-15.
[29] John Nickolls, Ian Buck, Michael Garland, and Kevin Skadron. Scalable parallel programming
with cuda. In ACM SIGGRAPH 2008 Classes, SIGGRAPH ’08, New York, NY, USA, 2008.
Association for Computing Machinery.
[30] Allen Nie, Yi Su, Bo Chang, Jonathan N. Lee, Ed H. Chi, Quoc V. Le, and Minmin Chen.
Evolve: Evaluating and optimizing llms for exploration, 2024.
[31] NVIDIA. Gpu mode at nvidia gtc 2025, 2025. Accessed: 2025-05-15.
[32] NVIDIA Corporation. Cutlass: Cuda templates for linear algebra subroutines, May 2025.
Accessed: 2025-05-15.
[33] Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and
Azalia Mirhoseini. Kernelbench: Can llms write efficient gpu kernels?, 2025.
[34] Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching,
Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement
fine-tuning, 2025.
[35] Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, and Arian Hosseini.
Putting the value back in rl: Better test-time scaling by unifying llm reasoners with verifiers,
2025.
[36] Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, and Razvan Pascanu.
Llms are greedy agents: Effects of rl fine-tuning on decision-making abilities, 2025.
[37] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms, 2017.
[38] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of
mathematical reasoning in open language models, 2024.
[39] Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob Gardner, Milad Hashemi,
Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, and Amir Yazdanbakhsh. Learning
performance-improving code edits, 2024.
[40] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur
Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap,
Karen Simonyan, and Demis Hassabis. Mastering chess and shogi by self-play with a general
reinforcement learning algorithm, 2017.
[41] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute
optimally can be more effective than scaling model parameters, 2024.
[42] Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, and Christopher Ré. Thun-
derkittens: Simple, fast, and adorable ai kernels, 2024.
[43] Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff
Schneider, and Ruslan Salakhutdinov. Training a generally curious agent, 2025.
[44] Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
[45] NovaSky Team. Sky-t1: Train your own o1 preview model within $450. https://novasky-
ai.github.io/posts/sky-t1, 2025. Accessed: 2025-01-09.
[46] Prime Intellect Team, Sami Jaghouar, Justus Mattern, Jack Min Ong, Jannik Straube, Manveer
Basra, Aaron Pazdera, Kushal Thaman, Matthew Di Ferrante, Felix Gabriel, Fares Obeid, Kemal
Erdem, Michael Keiblinger, and Johannes Hagemann. Intellect-2: A reasoning model trained
through globally decentralized reinforcement learning, 2025.
[47] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
[48] Philippe Tillet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for
tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International
Workshop on Machine Learning and Programming Languages, MAPL 2019, page 10–19, New
York, NY, USA, 2019. Association for Computing Machinery.
[49] Siddhant Waghjale, Vishruth Veerendranath, Zora Zhiruo Wang, and Daniel Fried. Ecco: Can
we improve model-generated code efficiency without sacrificing functional correctness?, 2024.
[50] Guanhua Wang, Heyang Qin, Sam Ade Jacobs, Connor Holmes, Samyam Rajbhandari, Olatunji
Ruwase, Feng Yan, Lei Yang, and Yuxiong He. Zero++: Extremely efficient collective commu-
nication for giant model training, 2023.
[51] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi
Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang
Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin,
Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai
software developers as generalist agents, 2025.
[52] Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng,
Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and
Yelong Shen. Reinforcement learning for reasoning in large language models with one training
example, 2025.
[53] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan
Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho,
Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding
self-evolution in llm agents via multi-turn reinforcement learning, 2025.
[54] Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel
Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I. Wang. Swe-rl: Advancing llm reasoning
via reinforcement learning on open software evolution, 2025.
[55] Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy
Phung, Rafael Rafailov, Nathan Lile, Dakota Mahan, Louis Castricato, Jan-Philipp Franken,
Nick Haber, and Chelsea Finn. Towards system 2 reasoning in llms: Learning how to think with
meta chain-of-thought, 2025.
[56] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen,
Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. Flashinfer: Efficient and
customizable attention engine for llm inference serving. arXiv preprint arXiv:2501.01005,
2025.
[57] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, ZhiqiLin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie
Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou,
Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan
Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
[58] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao
Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the
base model?, 2025.
[59] Chenggang Zhao, Liang Zhao, Jiashi Li, and Zhean Xu. Deepgemm: clean and efficient fp8
gemm kernels with fine-grained scaling. https://github.com/deepseek-ai/DeepGEMM,
2025.
[60] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for
formal olympiad-level mathematics, 2022.
[61] Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training
language model agents via hierarchical multi-turn rl, 2024.
[62] Richard Zhuang*, Trung Vu*, Alex Dimakis, and Maheswaran Sathiamoorthy. Improving
multi-turn tool use with reinforcement learning. https://www.bespokelabs.ai/blog/improving-
multi-turn-tool-use-with-reinforcement-learning, 2025. Accessed: 2025-04-17.
## A KernelBench Modifications

We use KernelBench [33] as our training environments. KernelBench is a popular benchmark for
evaluating LLMs’ ability to generate performant CUDA kernels for deep learning workloads in
PyTorch. Each KernelBench task consists in generating a CUDA kernel given a PyTorch reference
implementation, which is used to evaluate correctness and speedup.
### A.1 Task Improvements

We identify several limitations in the original KernelBench and introduce targeted modifications to
address them. These changes are crucial to mitigate reward hacking, as shown in Section 6.2.
- We sand-boxed the kernel evaluation process so that fatal errors, such as CUDA illegal
memory accesses, do not crash the RL training process.
- A significant issue we noted in KernelBench was that for many tasks, the input tensors used
to measure performance are quite small. This causes kernel launch overhead to take up a
significant portion of the runtime. To address this, we enlarged the tensor dimensions of the
affected tasks.
- A sneakier bug in the KernelBench’s evaluation harness caused the tested kernel to recycle
the output tensor from the reference implementation (which was run immediately before) as
its own tensor output. As a result of this, a kernel that only computes (correctly) a portion
of the output tensor would still pass the correctness check. We address this by running the
tested kernel first and only after the reference implementation, thus avoiding this hack.
In the end, we chose a total of 180 tasks as training environments, with 90 of the 100 Level 1 problems
and 90 Level 2 problems (sequences of operators with fusion opportunities).
### A.2 Construction of Additional Evaluation Set

Since current KernelBench does not provide a train-test split, we construct 80 additional tasks
following the same methodology that KernelBench was constructed.
KernelBench Level 2 is constructed by composing a subset of PyTorch operators as sequences of
operators. Specifically, the PyTorch operators are categorized as:
- Main operators: Conv2d, Matmul, Gemm, BMM, Conv3d, ConvTranspose2d,
ConvTranspose3d.
- Activations: ReLU, Sigmoid, Tanh, LeakyReLU, GELU, Swish, Softmax, Mish,
Hardtanh, HardSwish.
- Element-wise operators: Add, Multiply, Subtract, Divide, Clamp, Scale,
ResidualAdd.
- Normalizations: BatchNorm, LayerNorm, InstanceNorm, GroupNorm.
- Pooling: MaxPool, AvgPool, GlobalAvgPool.
- Bias: BiasAdd.
- Reductions: Sum, Mean, Max, Min, LogSumExp.
- Others: ResidualAdd, Scaling.
To construct the additional eval set (unseen from train set), following the methodology from original
KernelBench task construction:
1. We sample from the available operators listed above: 1 main operator (computationally
expensive), and 2-5 other operators.
2. We ask a language model, namely Gemini 2.5-Flash [10], to generate a PyTorch program
that creates a kernel by combining these operators. We also ask it to generate sample tensor
sizes for the task.
3. We ensure this PyTorch program can be executed and has a runtime on NVIDIA H200
> 0.1ms, to avoid the runtime being dominated by kernel launch (CPU) overhead.
4. We make sure this PyTorch program (with the same sequence of operators) is not present in
existing KernelBench Level 1 and 2 programs.
We manually inspected all new task programs to ensure their validity. We build the evaluation set by
combining our 80 newly created tasks with the 20 remaining original KernelBench tasks, for a total
of 100 unseen evaluation tasks.
## B Additional Details on Multi-Turn RL

Here we elaborate on design choices for our RL Training as described in Section 3.3 and Section 4,
along with some ablation results.
### B.1 Motivation for Turn-wise Reward

In our multi-turn RL training setup, within each training step we have a trajectory with n refinement
turns. A possible approach would be to compute the reward based on the kernel at the last turn,
similar to what is used in RLEF [13]. However, for the GPU kernel optimization setting, using just
the last kernel might not be optimal at times: for example, as shown earlier in Figure 1, kernel 3 is
correct but kernel 4 is incorrect as the model attempts more aggressive optimizations.
In this setting, computing reward based on the best kernel among the trajectory instead (max speedup)
is a more natural choice. However, using only the max kernel score forces us to discard all turns in
a trajectory after the max turn, possibly wasting a significant amount of inference rollouts: In the
previous example, we would have to completely discard the reasoning trace, code, and evaluation for
kernel 4. Thus, we arrived at our approach in Section 4.3, which uses a discounted look-ahead max
or sum, enabling more sample-efficient training.

**==> picture [Figure 8] intentionally omitted <==**

Figure 8: Training reward with correctness
weighting of 1, performance / speedup weighting
of1. Concretely, S = 1{correct}+Tbaseline
Tkernel
·1{correct}
.

**==> picture [Figure 9] intentionally omitted <==**

Figure 9: Training reward with no correctness
weighting, performance / speedup weighting of 1.
(speedup is 0 if kernel is incorrect). Concretely,
S = 1{correct} · Tbaseline
Tkernel
.
### B.2 Weighting for Score

In Section 3.2, we explain our score design, which assigns a scalar value (score S) based on a kernel’s
correctness and speedup. We explore score design and how to balance the correctness-performance
trade-off, after series of small-scale ablations on QwQ-32B [47].
We decided on a weighting of 0.3 on correctness and using speedup for performance (raw speedup
itself, no weighting), which is S = 0.3 · 1{correct} + 1{correct} · Tbaseline
Tkernel
.
Here we present some ablation studies we ran with different weighting configurations for score
design, particularly focusing on adjusting the weighing for correctness, in the context of single-turn
RL (GRPO) training (as shown in Section 3.3). As show an example in Figure 8, where we set
the weighting to 1.0 for correctness, the reward plateaus and eventually decreased; concretely, we
observed that the model over-optimizes for generating correct kernels and does not explore speedup
as much, causing the reward to plateau during training. In another experiment in Figure 9, we set
the weighting to 0 for correctness, only rewarding the model for generating performant (and correct)
kernels. We again observed the reward plateau. Thus, we hypothesize that it is still important to
reward the model for correct kernels, as long as the correctness reward is not too significant, balancing
the correctness-performance tradeoff.
### B.3 Number of Trajectories during Training

We vary the number of parallel trajectories during Multi-Turn RL training (Section 4), using 64
parallel trajectories instead of 16 for each task during each training step. We note that best@16
correctness slightly increases, but the overall performance does not show significant improvements.
Due to the high-compute requirements of doing more generations during training, we chose to train
with 16 parallel trajectories.
### B.4 Length Penalty

We explore incorporating response length as a part of the reward design to incentivize the model to
use its reasoning tokens more efficiently. We attempted a run using the length penalty from Kimi [44]
on DeepSeek-R1-Distill-Qwen-14B. As shown in Figures 10 and 11, we found that the response
length of the responses collapses, with the model no longer outputting CoT after 10 training steps,
suggesting that the addition of a length penalty is counterproductive for our setting.

**==> picture [Figure 10] intentionally omitted <==**

Figure 10: Training Reward collapses when in-
cluding length penalty as part of reward

**==> picture [Figure 11] intentionally omitted <==**

Figure 11: Response length of generations col-
lapses when including length penalty as part of
reward.
## C RL Infrastructure

Although a few open-source RL frameworks existed when we began this study, it is still difficult to
support training in a kernel evaluation environment and including multiple turns within one training
step. We built our training framework on top of the OpenRLHF [16] framework.
We use vLLM [19] for inference and DeepSpeed Zero-3 [50] for offloading optimizer states.
Each of the 8 GPUs handles the kernel generation and evaluation for one task. After the response
generation finishes, each GPU offloads its vLLM engine to CPU memory and evaluates the kernels it
generated. We run the evaluation and calculate reward and evaluation info. Each GPU then wakes up
its corresponding vLLM engine and regenerates kernels.
Each full RL training run took multiple days due to the limited compute we have. Hence to iterate
quickly and compare across configurations, we train up to 40-50 global steps (80-100 gradient steps).

**==> picture [Figure 12] intentionally omitted <==**

Figure 12: Overview of our RL Training infrastructure.
## D Inference Setup

Our prompt is similar to the prompt used in KernelBench [33]. We use this during training and
test-time inference. In the first refinement turn, we add an example of the inline CUDA format to the
prompt but remove it afterwards.
Below we show how we construct the context in the simplest case (of one turn, or the base prompt).
In the context, we present model the KernelBench task, instructions, and a simple 1-shot example of
a CUDA add kernel (to inform model the desired format for response):
1 You are given the following architecture:
2 import torch
3 import torch.nn as nn
4
5 class Model(nn.Module):
6 """
7 Simple model that performs Layer Normalization.
8 """
9 def __init__(self, normalized_shape: tuple):
10 """
11 Initializes the LayerNorm layer.
12
13 Args:
14 normalized_shape (tuple): Shape of the input tensor to be normalized.
15 """
16 super(Model, self).__init__()
17 self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
18
19 def forward(self, x: torch.Tensor) -> torch.Tensor:
20 """
21 Applies Layer Normalization to the input tensor.
22
23 Args:
24 x (torch.Tensor): Input tensor of shape (*, normalized_shape).
25
26 Returns:
27 torch.Tensor: Output tensor with Layer Normalization applied, same
shape as input.
28 """
29 return self.ln(x)
30
31 Replace pytorch operators in the given architecture with raw CUDA kernels,
optimizing for performance on NVIDIA H100 (e.g. shared memory, kernel fusion,
warp primitives, vectorization,...). Use torch.utils.cpp_extension.load_inline
and name your optimized output architecture ModelNew. You are not allowed to
use torch.nn (except for Parameter, containers, and init). The input and
output have to be on CUDA device. Your answer must be the complete new
architecture (no testing code, no other code): it will be evaluated and you
will be given feedback on its correctness and speedup so you can keep
iterating, trying to maximize the speedup. After your answer, summarize your
changes in a few sentences.Here is an example:
32
33 import torch.nn as nn
34 from torch.utils.cpp_extension import load_inline
35
36 # Define the custom CUDA kernel for element-wise addition
37 elementwise_add_source = """
38 #include <torch/extension.h>
39 #include <cuda_runtime.h>
40
41 __global__ void elementwise_add_kernel(const float* a, const float* b, float* out,
int size) {
42 int idx = blockIdx.x * blockDim.x + threadIdx.x;
43 if (idx < size) {
44 out[idx] = a[idx] + b[idx];
45 }
46 }
47
48 torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
49 auto size = a.numel();
50 auto out = torch::zeros_like(a);
51
52 const int block_size = 256;
53 const int num_blocks = (size + block_size - 1) / block_size;
54
55 elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(),
b.data_ptr<float>(), out.data_ptr<float>(), size);
56
57 return out;
58 }
59 """
60
61 elementwise_add_cpp_source = (
62 "torch::Tensor␣elementwise_add_cuda(torch::Tensor␣a,␣torch::Tensor␣b);"
63 )
64
65 # Compile the inline CUDA code for element-wise addition
66 elementwise_add = load_inline(
67 name="elementwise_add",
68 cpp_sources=elementwise_add_cpp_source,
69 cuda_sources=elementwise_add_source,
70 functions=["elementwise_add_cuda"],
71 verbose=True,
72 extra_cflags=[""],
73 extra_ldflags=[""],
74 )
75
76
77 class ModelNew(nn.Module):
78 def __init__(self) -> None:
79 super().__init__()
80 self.elementwise_add = elementwise_add
81
82 def forward(self, a, b):
83 return self.elementwise_add.elementwise_add_cuda(a, b)
For our multi-turn RL training (Section 4) and inference (Section 5), we provide model with the
kernels, CoTs (summarized), and evaluation results of all previous turns in chronological order. We
truncate the turns that do not fit inside the context window, starting from the earliest ones.
1 <Base prompt containing pytorch architecture and instruction>
2
3 Here are your previous attempts:
4
5 < for each (i) previously generated kernel >
6 <Previously generated kernel G[i]>
7
8 <Summary of CoT[i]>
9
10 <if parsing error>
11
12 Your previous answer failed to be parsed due to not adhering to the desired
formatting. Here is the error message: <error_message>
13
14 <elif compilation error>
15
16 Your previous answer failed to compile. Here is the error message:
<error_message>
17
18 <elif run error>
19
20 Your previous answer compiled successfully but had runtime errors. Here is
the error message: <error_message>
21
22 <elif correctness error>
23
24 Your previous answer was incorrect. Here is the error message:
<error_message>
25
26 <elif correct>
27
28 Your previous answer was correct but can be made faster. Here is the
speedup you achieved relative to the baseline: <speedup>
29
30 Restart your reasoning process and generate new, complete code.
## E Training Stability

The analysis of the "not okay ratio" led us to believe that model instability caused the appearance of
nonsensical and repetitive outputs. Therefore, we attempted runs where we enabled KL divergence
penalty in the GRPO loss, which would penalize the model from deviating from the base policy too
much. Following DeepScaleR [27], we set the KL coefficient to 0.001 and attempted an ablation run.
However, we found that the reward plateaus with KL enabled, suggesting that the KL penalty slows
down learning. Thus we attempted other techniques of constraining the model from deviating into
regions of instability, such as clipping the gradient norm aggressively — which was effective in our
setting.

**==> picture [Figure 13] intentionally omitted <==**

Figure 13: Adding a KL penalty slows down learning. Here we conduct an ablation with KL
coefficient β = 0.001 versus β = 0. We see that the reward plateaus with KL enabled.
We use 4 refinement turns at train-time for efficient training. During test time, we can afford more
extensive test-time compute, so we evaluate on 8 turns instead of 4 turns.
## F Reward Hacking

Here we present excerpts from generated kernels that show signs of reward hacking, previously
mentioned in Section 6.2.
In the following example, the model simply copies the PyTorch reference implementation, thus
getting rewarded for generating a correct answer with 1.0x speedup. To prevent this, we modify
our kernel evaluation environment so that it checks each generated kernel if it contains instances of
torch.nn or torch.nn.functional. We assign a reward of 0 to those.
1 class ModelReLU(Module):
2 ...
3 def forward(self, x):
4 relu = torch.nn.ReLU()
5 return relu(x)
Similarly, the model wraps an incorrect implementation of the CUDA kernel in a try-except statement
and invokes the PyTorch implementation functions as a fallback. To prevent this, we assign a reward
of 0 to kernels that contain try or except.
1 class ModelReLU(Module):
2 ...
3 def forward(self, x):
4 try:
5 ... \# CUDA implementation
6 except Exception as e:
7 print("Custom␣ReLU␣kernel␣failed␣to␣compile.␣Using␣default␣ReLU␣
instead.")
8 relu = torch.nn.ReLU()
9 return relu(x)
Here the model inherits from the reference implementation, bypassing the need for a CUDA imple-
mentation. To prevent this, we assign a reward of 0 to kernels that contain pass.
1 class ModelReLU(Module):
2 ...
3 def forward(self, x):
4 ... \# CUDA implementation
5
6 class ModelReLU(Model): \# Model is the the reference implementation
7 pass
## G Kernel Highlight

As an example, we look at one the task form our evaluation set, Conv3D_GELU_Tanh_Clamp:
1 import torch
2 import torch.nn as nn
3
4 class Model(nn.Module):
5 """
6 Model using Conv3d, GELU, Clamp, Tanh
7 """
8 def __init__(self, in_channels, out_channels, kernel_size, clamp_min,
clamp_max):
9 super(Model, self).__init__()
10 self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
11 self.clamp_min = clamp_min
12 self.clamp_max = clamp_max
13
14 def forward(self, x):
15 """
16 Forward pass of the model.
17
18 Args:
19 x (torch.Tensor): Input tensor of shape (batch_size, in_channels,
depth, height, width).
20
21 Returns:
22 torch.Tensor: Output tensor of shape (batch_size, out_channels,
depth-kernel_size+1, height-kernel_size+1, width-kernel_size+1).
23 """
24 x = self.conv(x)
25 x = torch.tanh(x)
26 x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
27 x = torch.nn.functional.gelu(x)
28 return x
The task is made up of a computationally intensive function (3D convolution) and a series of
lightweight functions (tanh, clamp, gelu).
The model implements a correct solution in the first turn by fusing the kernels, but it only achieves a
speedup of 1.06x over the reference implementation.
1 import torch
2 from torch.utils.cpp_extension import load_inline
3 import math
4
5 class ModelNew(torch.nn.Module):
6 def __init__(self, in_channels, out_channels, kernel_size, clamp_min,
clamp_max):
7 super(ModelNew, self).__init__()
8 self.clamp_min = clamp_min
9 self.clamp_max = clamp_max
10
11 # Initialize convolution weights and bias
12 self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels,
kernel_size, kernel_size, kernel_size))
13 self.bias = torch.nn.Parameter(torch.empty(out_channels))
14 # Weight initialization following PyTorch’s Conv3d default
15 torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
16 fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
17 bound = 1 / math.sqrt(fan_in)
18 torch.nn.init.uniform_(self.bias, -bound, bound)
19
20 # Define the fused convolution and activation CUDA kernel
21 conv_activation_source = """
22 #include <torch/extension.h>
23 #include <cuda_runtime.h>
24 #include <math.h>
25
26 __global__ void conv_activation_kernel(const float* input, const float*
weights, const float* bias, float* output,
27 int N, int C_in, int D, int H, int W,
int C_out, int kD, int kH, int kW,
28 float clamp_min, float clamp_max) {
29 int idx = blockIdx.x * blockDim.x + threadIdx.x;
30 const int W_out = W - kW + 1;
31 const int H_out = H - kH + 1;
32 const int D_out = D - kD + 1;
33 const int total_elements = N * C_out * D_out * H_out * W_out;
34 if (idx >= total_elements) return;
35
36 // Decompose index into output coordinates
37 int w_out = idx % W_out;
38 int h_out = (idx / W_out) % H_out;
39 int d_out = (idx / (W_out * H_out)) % D_out;
40 int f = (idx / (W_out * H_out * D_out)) % C_out;
41 int n = idx / (C_out * D_out * H_out * W_out);
42
43 float sum = 0;
44 for (int c = 0; c < C_in; ++c) {
45 for (int kd = 0; kd < kD; ++kd) {
46 for (int kh = 0; kh < kH; ++kh) {
47 for (int kw = 0; kw < kW; ++kw) {
48 // Input tensor index calculation
49 int d_in = d_out + kd;
50 int h_in = h_out + kh;
51 int w_in = w_out + kw;
52 int input_idx = n * C_in * D * H * W +
53 c * D * H * W +
54 d_in * H * W +
55 h_in * W +
56 w_in;
57 // Weight tensor index calculation
58 int weight_idx = f * C_in * kD * kH * kW +
59 c * kD * kH * kW +
60 kd * kH * kW +
61 kh * kW +
62 kw;
63 sum += input[input_idx] * weights[weight_idx];
64 }
65 }
66 }
67 }
68 sum += bias[f];
69
70 // Apply activations sequentially
71 sum = tanh(sum);
72 sum = fmaxf(clamp_min, fminf(sum, clamp_max));
73 {
74 float inner = sum + 0.044715f * sum * sum * sum;
75 inner *= 0.79788456f; // sqrt(2/pi) approximation
76 sum = sum * 0.5f * (1.0f + tanh(inner));
77 }
78
79 // Output tensor index calculation
80 int output_idx = n * C_out * D_out * H_out * W_out +
81 f * D_out * H_out * W_out +
82 d_out * H_out * W_out +
83 h_out * W_out +
84 w_out;
85 output[output_idx] = sum;
86 }
87
88 torch::Tensor conv_activation_cuda(torch::Tensor input, torch::Tensor
weights, torch::Tensor bias,
89 float clamp_min, float clamp_max) {
90 // Dimension extraction
91 int N = input.size(0);
92 int C_in = input.size(1);
93 int D = input.size(2);
94 int H = input.size(3);
95 int W = input.size(4);
96 int C_out = weights.size(0);
97 int kD = weights.size(2);
98 int kH = weights.size(3);
99 int kW = weights.size(4);
100
101 // Output dimensions
102 int D_out = D - kD + 1;
103 int H_out = H - kH + 1;
104 int W_out = W - kW + 1;
105
106 // Create output tensor
107 auto output = torch::empty({N, C_out, D_out, H_out, W_out},
input.options());
108
109 // Launch kernel
110 int total_elements = N * C_out * D_out * H_out * W_out;
111 const int block_size = 256;
112 const int grid_size = (total_elements + block_size - 1) / block_size;
113 conv_activation_kernel<<<grid_size, block_size>>>(
114 input.data_ptr<float>(), weights.data_ptr<float>(),
bias.data_ptr<float>(),
115 output.data_ptr<float>(), N, C_in, D, H, W, C_out, kD, kH, kW,
116 clamp_min, clamp_max
117 );
118 return output;
119 }
120 """
121
122 conv_activation_header = """
123 torch::Tensor conv_activation_cuda(torch::Tensor input, torch::Tensor
weights, torch::Tensor bias,
124 float clamp_min, float clamp_max);
125 """
126
127 # Load the CUDA kernel
128 self.conv_activation = load_inline(
129 name="conv_activation",
130 cpp_sources=conv_activation_header,
131 cuda_sources=conv_activation_source,
132 functions=["conv_activation_cuda"],
133 verbose=True
134 )
135
136 def forward(self, x):
137 return self.conv_activation.conv_activation_cuda(x, self.weight, self.bias,
self.clamp_min, self.clamp_max)
On turn 2 it attempts to improve on the existing implementation, but fails due to a compilation error.
On turn 3, it attempts to modify the loop order to improve coalesced memory access, but this actually
worsens performance, as the kernel now exhibits a speedup of 0.61x. The model attempts this strategy
again at turn 4, and is able to order the loops in a way that increases memory coalescing: this brings
up the performance to 1.19x. At turn 5 and 6, the model attempts some minor optimizations (such as
precomputing certain indices, or deleting intermediate variables), which however has no measurable
effect in performance.
On turn 7, the model figures out a surprising optimization: it records the convolution’s kernel size
as a compile time constant, which allows it to use # pragma unroll to unroll the 3 loops needed to
apply the convolution. This results in a significant speedup of 1.93x over baseline.
1 from torch.utils.cpp_extension import load_inline
2 import math
3
4 class ModelNew(torch.nn.Module):
5 def __init__(self, in_channels, out_channels, kernel_size, clamp_min,
clamp_max):
6 super(ModelNew, self).__init__()
7 self.clamp_min = clamp_min
8 self.clamp_max = clamp_max
9
10 # Initialize convolution weights and bias
11 self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels,
kernel_size, kernel_size, kernel_size))
12 self.bias = torch.nn.Parameter(torch.empty(out_channels))
13 # Weight initialization following PyTorch’s Conv3d default
14 torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
15 fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
16 bound = 1 / math.sqrt(fan_in)
17 torch.nn.init.uniform_(self.bias, -bound, bound)
18
19 # Define the fused convolution and activation CUDA kernel with optimizations
20 kernel_size_val = kernel_size
21 conv_activation_source = f"""
22 #include <torch/extension.h>
23 #include <cuda_runtime.h>
24 #include <math.h>
25
26 #define KERNEL_SIZE {kernel_size_val}
27 #define KERNEL_D {kernel_size_val}
28 #define KERNEL_H {kernel_size_val}
29 #define KERNEL_W {kernel_size_val}
30
31 __global__ void conv_activation_kernel(
32 const float* __restrict__ input, const float* __restrict__ weights,
const float* __restrict__ bias,
33 float* __restrict__ output,
34 int N, int C_in, int D, int H, int W, int C_out,
35 float clamp_min, float clamp_max) {{
36 int idx = blockIdx.x * blockDim.x + threadIdx.x;
37 const int W_out = W - KERNEL_W + 1;
38 const int H_out = H - KERNEL_H + 1;
39 const int D_out = D - KERNEL_D + 1;
40 const int total_elements = N * C_out * D_out * H_out * W_out;
41 if (idx >= total_elements) return;
42
43 int w_out = idx % W_out;
44 int h_out = (idx / W_out) % H_out;
45 int d_out = (idx / (W_out * H_out)) % D_out;
46 int f = (idx / (W_out * H_out * D_out)) % C_out;
47 int n = idx / (C_out * D_out * H_out * W_out);
48
49 float sum = 0.0;
50
51 // Precompute constants for GELU
52 const float sqrt_2_over_pi = 0.79788456f;
53 const float approx_coeff = 0.044715f;
54
55 for (int c = 0; c < C_in; ++c) {{
56 // Compute input channel base pointer
57 const int input_c_offset = n * C_in * D * H * W + c * D * H * W;
58 const float* input_c = input + input_c_offset;
59
60 // Compute center position in the input channel
61 const int center_offset = d_out * H * W + h_out * W + w_out;
62 const float* center = input_c + center_offset;
63
64 // Weight base for current output channel f and input channel c
65 const int weight_base = f * C_in * KERNEL_D * KERNEL_H *
KERNEL_W + c * KERNEL_D * KERNEL_H * KERNEL_W;
66
67 #pragma unroll
68 for (int kd = 0; kd < KERNEL_D; ++kd) {{
69 #pragma unroll
70 for (int kh = 0; kh < KERNEL_H; ++kh) {{
71 #pragma unroll
72 for (int kw = 0; kw < KERNEL_W; ++kw) {{
73 const int offset = kd * H * W + kh * W + kw;
74 const float input_val = center[offset];
75 const int w_off = weight_base + kd * KERNEL_H *
KERNEL_W + kh * KERNEL_W + kw;
76 const float weight_val = weights[w_off];
77
78 sum += input_val * weight_val;
79 }}
80 }}
81 }}
82 }}
83
84 sum += bias[f];
85
86 // Apply activations with optimized order
87 {{
88 float tanh_val = tanhf(sum);
89 float clamped = fmaxf(clamp_min, fminf(tanh_val, clamp_max));
90 float inner = clamped + approx_coeff * clamped * clamped *
clamped;
91 inner *= sqrt_2_over_pi;
92 sum = clamped * 0.5f * (1.0f + tanhf(inner));
93 }}
94
95 // Output index calculation
96 const int output_base = n * C_out * D_out * H_out * W_out +
97 f * D_out * H_out * W_out +
98 d_out * H_out * W_out +
99 h_out * W_out;
100 output[output_base + w_out] = sum;
101 }}
102
103 torch::Tensor conv_activation_cuda(torch::Tensor input, torch::Tensor
weights, torch::Tensor bias,
104 float clamp_min, float clamp_max) {{
105 int N = input.size(0);
106 int C_in = input.size(1);
107 int D = input.size(2);
108 int H = input.size(3);
109 int W = input.size(4);
110 int C_out = weights.size(0);
111
112 int D_out = D - KERNEL_D + 1;
113 int H_out = H - KERNEL_H + 1;
114 int W_out = W - KERNEL_W + 1;
115
116 auto output = torch::empty({{N, C_out, D_out, H_out, W_out}},
input.options());
117
118 int total_elements = N * C_out * D_out * H_out * W_out;
119 const int block_size = 256;
120 const int grid_size = (total_elements + block_size - 1) / block_size;
121
122 conv_activation_kernel<<<grid_size, block_size>>>(
123 input.data_ptr<float>(), weights.data_ptr<float>(),
bias.data_ptr<float>(),
124 output.data_ptr<float>(), N, C_in, D, H, W, C_out,
125 clamp_min, clamp_max
126 );
127 return output;
128 }}
129 """
130
131 conv_activation_header = """
132 torch::Tensor conv_activation_cuda(torch::Tensor input, torch::Tensor
weights, torch::Tensor bias,
133 float clamp_min, float clamp_max);
134 """
135
136 # Load the CUDA kernel with fast math optimization
137 self.conv_activation = load_inline(
138 name="conv_activation",
139 cpp_sources=conv_activation_header,
140 cuda_sources=conv_activation_source,
141 functions=["conv_activation_cuda"],
142 extra_cuda_cflags=[’-use_fast_math’],
143 verbose=True
144 )
145
146 def forward(self, x):
147 return self.conv_activation.conv_activation_cuda(x, self.weight, self.bias,
self.clamp_min, self.clamp_max)
In its final turn, the model attempts a more advanced implementation that further parallelizes the
computation across kernels before performing a warp-level reduction. However, it fails to implement
the strategy correctly, due to applying the reduction across the wrong axis. We do note the model has
shown success in implementing complex warp reductions in several other tasks.
