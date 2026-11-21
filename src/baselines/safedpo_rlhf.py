"""`swift rlhf --rlhf_type dpo` with the SafeDPO safety margin (Kim et al., ICLR 2026, Eq. 12):

    L = -log sigma( beta * [ (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l)) ] - (h_l - h_w) * Delta )

The per-pair term (h_l - h_w) * Delta (Delta for a (safe, unsafe) pair, 0 otherwise) is read from the `margin` field of the pair file written by src/baselines/matched_pool_compute.py; ms-swift 4.5.3 already
carries a dataset `margin` column to the batch (reward-model training) but its DPO trainer would pass it to the model, so it is removed from the batch here and subtracted inside the sigmoid DPO loss.
Everything else (LoRA, optimiser, schedule, data order for a given seed, reference model = adapter switched off) is the unmodified ms-swift DPO trainer.  With margin = 0 for all pairs the loss is exactly the standard DPO loss.
usage: python src/baselines/safedpo_rlhf.py <the arguments of `swift rlhf`>
"""
import torch.nn.functional as F


def patch():
    from swift.rlhf_trainers.dpo_trainer import DPOTrainer

    if getattr(DPOTrainer, "_safedpo_patched", False):
        return
    original_batch = DPOTrainer.get_batch_loss_metrics
    original_loss = DPOTrainer.dpo_loss

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        batch = dict(batch)
        self._safedpo_margin = batch.pop("margin", None)
        try:
            return original_batch(self, model, batch, train_eval)
        finally:
            self._safedpo_margin = None

    def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_type="sigmoid", model_output=None):
        margin = getattr(self, "_safedpo_margin", None)
        if margin is None or loss_type != "sigmoid":
            return original_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_type, model_output)
        assert not self.reference_free and self.label_smoothing == 0 and self.f_divergence_type == "reverse_kl", "the SafeDPO margin is implemented for the plain sigmoid DPO loss"
        device = self.accelerator.device
        logits = (chosen_logps - rejected_logps).to(device) - (ref_chosen_logps - ref_rejected_logps).to(device)
        margin = margin.to(device=device, dtype=logits.dtype).view(-1)
        losses = -F.logsigmoid(self.beta * logits - margin)
        if not getattr(self, "_safedpo_logged", False):
            print(f"[safedpo] first batch: margin={margin.tolist()} loss={losses.tolist()} (loss without margin={(-F.logsigmoid(self.beta * logits)).tolist()})", flush=True)
            self._safedpo_logged = True
        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
        return losses, chosen_rewards, rejected_rewards

    DPOTrainer.get_batch_loss_metrics = get_batch_loss_metrics
    DPOTrainer.dpo_loss = dpo_loss
    DPOTrainer._safedpo_patched = True


if __name__ == "__main__":
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()
    patch()
    from swift.pipelines import rlhf_main

    rlhf_main()
