The paper's closed-model conditions. Each reuses the published fine-tunes
(their OpenAI ids are below) instead of training, and generates honest and
honeypot solutions separately per collusion strategy, as the paper did.
Closed models don't resample identically, so a rerun gives new samples; for
the published numbers, pass the published `.eval` files as `inputs`.
