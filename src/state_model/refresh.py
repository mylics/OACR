from pathlib import Path

import torch
import torch.distributed as distributed


def gather_rollouts(rollouts):
    rows = [rollout.to_dict() for rollout in rollouts]
    if not distributed.is_initialized():
        return rows
    gathered = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, rows)
    return [row for part in gathered for row in part]


def refresh_state_model(scorer, trainer, buffer, epochs, accelerator, output_dir):
    status = [None]
    if accelerator.is_main_process:
        try:
            metrics = trainer.fit(buffer.records(), epochs=epochs)
            scorer.model.save_pretrained(Path(output_dir), tokenizer=scorer.tokenizer)
            status[0] = {"metrics": metrics, "error": None}
        except Exception as exc:
            status[0] = {"metrics": None, "error": f"{type(exc).__name__}: {exc}"}
    if distributed.is_initialized():
        distributed.broadcast_object_list(status, src=0, device=accelerator.device)
    if status[0]["error"]:
        raise RuntimeError(status[0]["error"])
    if distributed.is_initialized():
        with torch.no_grad():
            for value in scorer.model.state_dict().values():
                transferred = value.to(accelerator.device).contiguous()
                distributed.broadcast(transferred, src=0)
                value.copy_(transferred.to(value.device))
    scorer.model.eval()
    scorer.model.requires_grad_(False)
    return status[0]["metrics"]
