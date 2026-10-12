## PPO Critic Explained

Oct 8, 2026

Tinker supports PPO’s policy loss. The integration gap is HiPT’s attached value head, not the receiver or PPO itself.

·

@Someone

## Reward is a grade; value is an expectation

The receiver grades the finished code’s message. The critic predicts the sender’s expected future reward from the current generation state.

| Component | Question it answers | Output |
| --- | --- | --- |
| Sender / actor | What code token should I generate next? | A distribution over next tokens |
|   | Receiver / decoder Which secret bit is present in this code? | Label probabilities, such as probability of bit 1 |
| Critic / value estimator | Given this problem, bit, and code prefix, how much reward should I expect? | A scalar expected return |

Suppose a finished solution gets reward 0.8, while the critic expected 0.6. Its simplified advantage is +0.2: better than expected.

```
\# Teaching example for a terminal-reward problem, not HiPT's full estimator. advantage = observed_reward - predicted_value
```

Real token-level PPO can use a more detailed advantage estimator, including discounted rewards, KL penalties, and value predictions along the sequence. The simple subtraction explains the critic’s role.

The receiver’s correct-bit probability can contribute to observed reward. It does not automatically equal expected future return from every unfinished code prefix. APPS correctness adds another reward component.

Important correction to the earlier wording: a learned critic is not mandatory for every implementation of PPO’s clipped policy objective. It is part of HiPT’s particular reference implementation. Other advantage baselines are possible, but they change that implementation.


## The receiver grades code; the value head predicts return

The two sender outputs share a backbone in HiPT’s reference architecture. APPS tests are our proposed reward addition, not part of the original demographic experiment. This diagram shows the policy-update signal path; the value-head training loss and separate receiver-SFT updates are shown in the code sections below.

## What the paper actually commits to

Section 3.3.1 specifies alternating supervised receiver training and PPO–LoRA sender training. It does not explicitly require a particular critic architecture.

*[From §3.3.1, Gradient-based Reinforcement Learning:](https://arxiv.org/html/2410.03768v1#S3.SS3.SSS1)*

training of the sender and training of the receiver.”


```
training of the sender and training of the receiver.”
```

“The receiver is a pretrained language model with a randomly initialized classification head.”

“We use proximal policy optimization (PPO) … with low-rank adaptation (LoRA) … to update the sender”

Ellipses in the last passage omit the paper’s bibliographic citations, not a change in the method.

The same section defines sender reward as the receiver’s probability for the correct label, minus an overseer penalty:

Here, the receiver supplies the task-reward signal. That is different from the sender’s value estimate used inside PPO training.

[Appendix A.1](https://arxiv.org/html/2410.03768v1#A1.SS1)

gives the alternating update algorithms and identifies the original receiver

as DistilBERT with a classification head.

Paper-level method versus exact source reproduction: the attached value head is visible in the repository and its TRL dependency, shown below. The paper’s short section does not itself state that this particular attachment must be preserved. We can implement PPO–LoRA and the alternating learning mechanism without claiming an identical reproduction of every source-code detail.

## Where the value head appears in the code

HiPT’s sender has two outputs from one backbone: token predictions and scalar value predictions. The value head itself is small; the integration issue is access to hidden states and joint training, not the size of that head.

Exact HiPT excerpt — sender construction:

```
self.enc_model = AutoModelForCausalLMWithValueHead.from_pretrained(
self.enc_ppo_config.model_name,
peft_config=lora_config,
quantization_config=quantization_config,
attn_implementation=self.hparams['enc']['attn_implementation'],
torch_dtype=dtype,
```


```
)
```

The important part is WithValueHead . The repository also creates a reference model for

policy regularization.

[Source, lines 283–292.](https://github.com/olliematthews/lasr-steganography/blob/d4c708b9339fd2f85cb442827ac56333b55cf9d2/lasr-steganography/learning/rl/stego_trainer.py#L283-L292)

Exact TRL 0.9.6 excerpt — the extra scalar head:

```
self.summary = nn.Linear(hidden_size, 1)
```

This layer turns a hidden-state vector into one value. During the sender’s forward pass, TRL applies it to the backbone’s hidden states:

```
last_hidden_state = base_model_output.hidden_states[-1]
lm_logits = base_model_output.logits
loss = base_model_output.loss
if last_hidden_state.device != self.v_head.summary.weight.device:
last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
value = self.v_head(last_hidden_state).squeeze(-1)
```

Linear head, line 46; forward pass, lines 179–186. This is the dependency version listed in HiPT’s pinned environment. [URL 🔗](https://github.com/huggingface/trl/blob/v0.9.6/trl/models/modeling_value_head.py#L179-L186)

Exact HiPT excerpt — hand rewards to the full PPO trainer:

```
stats = self.enc_ppo_trainer.step(
list(batch["input_ids"]), list(batch["response_tensors"]), rewards_list
)
```

Source, lines 1321–1323. The TRL trainer handles value predictions and advantage estimation inside this call. [URL 🔗](https://github.com/olliematthews/lasr-steganography/blob/d4c708b9339fd2f85cb442827ac56333b55cf9d2/lasr-steganography/learning/rl/stego_trainer.py#L1321-L1323)

Exact TRL excerpts — use the values and train the value head:

```
delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
advantages_reversed.append(lastgaelam)
loss = pg_loss + self.config.vf_coef * vf_loss
```

Advantage calculation, lines 1173–1183; policy plus value loss, lines 1220–1233. [URL 🔗](https://github.com/huggingface/trl/blob/v0.9.6/trl/trainer/ppo_trainer.py#L1173-L1183)

values are learned predictions. returns provide training targets for those predictions. pg_loss updates the policy; vf_loss trains value prediction. These are source excerpts, not a standalone runnable training script.


## What Tinker does, and what we must supply

Tinker implements the clipped PPO policy loss. Its API expects our code to supply rollout log probabilities and advantages.

The documented PPO inputs are target_tokens , logprobs , and advantages . The returned log probabilities are token probabilities, not the scalar values from HiPT’s attached head. PPO documentation. [URL 🔗](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/)

Illustrative API usage — valid interface shape, not a complete training script:

```
\# Our rollout/advantage code prepares these aligned arrays.
loss_fn_inputs = {
"target_tokens": target_token_ids,
"logprobs": rollout_logprobs,
"advantages": computed_advantages,
}
# 'data' is a batch of Datum objects containing these inputs.
fwd_bwd_future = await training_client.forward_backward_async(
data=data,
loss_fn="ppo",
)
```

The key question is where computed_advantages comes from. Tinker does not fill in HiPT’s value head or GAE calculation merely because we select "ppo" .

Why a custom loss is not automatically an attached critic

The current TrainingClient contract describes loss_type_input as: [URL 🔗](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)

“Currently the only supported value is "logprobs".”

A custom loss can differentiate through remote token log probabilities. The documented interface does not expose the sender’s hidden states for attaching an arbitrary shared- backbone scalar head.

```
\# Conceptual illustration of the original local-PyTorch architecture.
# Not an operation offered by Tinker's documented custom-loss API.
value = scalar_head(sender_hidden_states)
```

Therefore, the original AutoModelForCausalLMWithValueHead object and its joint policy/

value loss are not a drop-in Tinker configuration.

This is an API-integration gap, not a ban on critics. We can compute advantages using


This is an API-integration gap, not a ban on critics. We can compute advantages using

another baseline, or implement a separate learned critic. A critic represented as another text-predicting model would also require a specific design and validation. None is automatically provided by changing the receiver to an LLM.

The statements above concern the public interfaces checked on the document’s as-of date. They do not claim that no future or private interface could expose additional model outputs.

## What this means for our APPS implementation

We can keep PPO on Tinker. We must choose an advantage estimator explicitly instead of assuming the API includes HiPT’s entire trainer.

There are two independent choices:

- 1. Policy loss: clipped PPO or another policy-gradient objective.

- 2. Advantage estimator: learned-value/GAE machinery or a baseline such as a group mean.

Grouping several candidates does not inherently rule out PPO. Those group-based advantages can be supplied to a clipped PPO policy loss. My earlier cookbook-default proposal used importance sampling, which would change the policy loss as well as the baseline.

| Implementation | Where the values or baseline come from | What differs from HiPT’s source |
| --- | --- | --- |
| Native PyTorch/TRL actor with attached value head | A scalar head on the sender’s shared backbone | Most direct route to reproducing its actor–critic structure; APPS still needs adaptation |
| Tinker PPO actor + separate learned critic | Our critic implementation supplies token-level values and advantages | Different critic placement/ sharing; does not automatically reproduce the original joint policy/value loss |
| Tinker PPO actor + group- based baseline | Rewards of multiple candidates for the same problem–payload pair | No learned value head; different advantage estimation |

The first option is closest to the reference implementation, not a promise of identical results. Changing the dataset, sender, receiver, or reward already changes the experiment.


experiment.

Replacing DistilBERT with a Tinker LLM only changes the receiver. It simplifies receiver infrastructure and lets us obtain payload-label probabilities. It does not automatically create a prediction of the sender’s future total reward.

For APPS, total reward must also account for executed-code correctness. The receiver’s ability to read a bit from a comment is not evidence that the code solves the problem.

Bottom line: the paper-level PPO–LoRA method is feasible on Tinker. Exact reproduction of the repository’s attached value head is not a documented drop-in feature. That is an engineering choice to resolve, not a reason to say “we cannot use §3.3.1.”

The code excerpts are explanatory; no model calls or training were run. No training design or paid experiment is approved by this document.

## The math: HiPT/TRL PPO vs Tinker PPO

Both use the same clipped policy objective, built from log-probs. They differ only in where the advantage comes from, and in whether a value loss is trained alongside. Values and log-probs are not alternatives: log-probs drive the policy ratio in both, and values are one way to build the advantage.

Shared part: the clipped policy loss. For token a_t in state s_t (prompt plus tokens so far), with log-probs from the current policy and from sampling time:

Tinker's logprobs input is log π_old(a_t | s_t), recorded when the stegotext was sampled. Tinker computes log π_θ itself and returns the gradient of this loss. In the call,

Â_t is the advantages array.

HiPT/TRL: advantage from the attached value head. The value head reads the last hidden state h_t, the same backbone that produces the logits:

The per-token reward folds in the KL penalty to the reference model, plus the episode

reward R (receiver score) on the last token T:


reward R (receiver score) on the last token T:

The advantage comes from GAE over those rewards and values, with γ = 1 and λ = 0.95 by default:

One backward pass trains the policy and the value head together, with the value target R̂_t = Â_t + V_φ(s_t):

The value loss is also clipped in TRL, and c_v = vf_coef . Because θ and φ share the backbone, the value loss also shapes the LoRA weights the policy uses.

Tinker with a group baseline: advantage from sibling samples. Sample K stegotexts for the same problem–payload pair, with rewards R_1 … R_K. Each sample's advantage is applied to all of its tokens:

The loss is L^clip alone, with no V_φ and no value term. If we want the KL penalty, we compute log π_ref ourselves and fold it into R_k or Â.

| Quantity | HiPT/TRL | Tinker "ppo" + group baseline |
| --- | --- | --- |
| Log-probs log π_old, log π_θ | Computed locally; used in ρ_t | We send log π_old; Tinker computes log π_θ |
| Values V(s_t) | Value head on hidden state h_t, per token | None |
| Advantage Â_t | GAE over per-token rewards and values; varies per token | Reward minus group mean; constant within a sample |
