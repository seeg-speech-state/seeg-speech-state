import typer
import os
import ast
import json
from typing import Annotated, Literal
from pathlib import Path
from enum import Enum
from loguru import logger
from tqdm import tqdm


app = typer.Typer()


@app.command()
def main(
    model_ksizes: Annotated[str, typer.Option(help="[7,5,5]")],
    model_filters: Annotated[str, typer.Option(help="[64,128,256]")],
    model_dropout: Annotated[float, typer.Option()],
    dataset_subject: Annotated[str, typer.Option()],
    dataset_sig_filter_used: Annotated[Literal["default", "high_gamma", "default_v2"], typer.Option()],
    dataset_num_classes: Annotated[int, typer.Option()],
    train_tag: Annotated[str, typer.Option(help="Training tag, used for saving model")],
    train_clear_old_ckpt: Annotated[bool, typer.Option()],
    train_batch_size: Annotated[int, typer.Option()],
    train_epochs: Annotated[int, typer.Option()],
    train_lr: Annotated[float, typer.Option()],
    train_weight_decay: Annotated[float, typer.Option()],
    train_metric: Annotated[Literal["BCE", "CE"], typer.Option()],
    train_optimizer: Annotated[Literal["Adam", "SGD"], typer.Option()],
    train_lr_sched_multistep: Annotated[str, typer.Option()] = "[]",
    train_checkpoint_dir: Annotated[str, typer.Option()] = "./checkpoints",
    train_eval_divisor: Annotated[int, typer.Option()] = 10,
    train_show_loss_divisor: Annotated[int, typer.Option()] = 100,
    dataset_dir: Annotated[str, typer.Option()] = "./dataset",
    dataset_split_way: Annotated[Literal["blender", "sentence"], typer.Option()] = "sentence",
    dataset_channels: Annotated[int, typer.Option()] = -1,
    dataset_channels_discard_ratio: Annotated[float | None, typer.Option()] = None,
    dataset_channel_config: Annotated[str | None, typer.Option()] = None,
    dataset_cat_samples: Annotated[bool, typer.Option()] = False,
    dataset_fast_dataloader: Annotated[bool, typer.Option()] = True,
    dataset_random_seed: Annotated[int, typer.Option()] = 42,
    dataset_drop_index: Annotated[str, typer.Option()] = "[]",
    debug_use_profiler: Annotated[bool, typer.Option()] = False
):
    cli_params = locals().copy()

    import torch
    import numpy as np
    from torch.utils.tensorboard.writer import SummaryWriter

    from dataset import Word300Dataset, collate_fn_time, SignalFilter
    from model import VanillaCNN1d, BatchNorm1d

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("main.log")

    assert torch.cuda.is_available(), "Must have CUDA"
    device = torch.device('cuda')
    dtype = torch.float32

    drop_index: list[str] = ast.literal_eval(dataset_drop_index)
    if len(drop_index) > 0:
        logger.info("dropping index {}", drop_index)

    dataset = Word300Dataset(
        Path(dataset_dir),
        dataset_subject,
        x_dtype=dtype,
        sig_filter_used=SignalFilter(dataset_sig_filter_used),
        drop_index=drop_index,
        channel_config=dataset_channel_config,
    )
    logger.info("using dataset={}", dataset)

    if dataset_channels_discard_ratio is not None and dataset_channels_discard_ratio > 0:
        logger.info("randomly discarding channels, ratio={}", dataset_channels_discard_ratio)
        dataset.random_discard_channels(dataset_channels_discard_ratio, dataset_random_seed)
    else:
        logger.debug("dataset_channels_discard_ratio not used")
        cli_params['dataset_channels_discard_ratio'] = None

    train_dataset, test_dataset = dataset.split(dataset_split_way, 0.8, dataset_random_seed)
    train_idx, test_idx = train_dataset.indices, test_dataset.indices
    logger.info("len(train_dataset)={}, len(train_idx)={}, len(test_dataset)={}, len(test_idx)={}", len(train_dataset), len(train_idx), len(test_dataset), len(test_idx))
    assert len(dataset) == len(train_dataset) + len(test_dataset)
    assert np.intersect1d(train_idx, test_idx, return_indices=False).size == 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        batch_size=train_batch_size,
        collate_fn=collate_fn_time if dataset_cat_samples else None,
        shuffle=True,
        prefetch_factor=4 if dataset_fast_dataloader else None,
        persistent_workers=dataset_fast_dataloader,  # https://github.com/pytorch/pytorch/issues/47445
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        batch_size=train_batch_size,
        collate_fn=collate_fn_time if dataset_cat_samples else None,
        shuffle=False,
        prefetch_factor=4 if dataset_fast_dataloader else None,
        persistent_workers=dataset_fast_dataloader,  # https://github.com/pytorch/pytorch/issues/47445
        drop_last=False,
    )

    output_layer = {"BCE": "sigmoid", "CE": "softmax"}[train_metric]
    ksizes: list[int] = ast.literal_eval(model_ksizes)
    filters: list[int] = ast.literal_eval(model_filters)

    if dataset_channels == -1:
        X, Y = dataset[0]
        dataset_channels = X.shape[0]
        logger.info("obtain dataset_channels={} from dataset", dataset_channels)
        cli_params['dataset_channels'] = dataset_channels

    model = VanillaCNN1d(
        in_channels=dataset_channels,
        cnn_ksizes=ksizes,
        block_filters=filters,
        fc_units=[],
        drop_out=model_dropout,
        classes=dataset_num_classes,
        act=torch.nn.ReLU,
        norm=BatchNorm1d,
        output_layer=output_layer,
        dtype=dtype,
    )
    model.to(device)
    logger.info("using model={}", model)

    if train_optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_lr, weight_decay=train_weight_decay
        )
    elif train_optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_lr,
            weight_decay=train_weight_decay,
            momentum=0.9,
            nesterov=False,
        )
    else:
        raise Exception(f"Unknown optimzer: {train_optimizer}")
    logger.info("using optimizer={}", optimizer)

    lr_sched_multistep = ast.literal_eval(train_lr_sched_multistep)
    if len(lr_sched_multistep) > 0:
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_sched_multistep, gamma=0.1
        )
    else:
        lr_sched = None
    logger.info("using lr_sched={}", lr_sched)

    # TODO: add reweight
    # metric_ce = torch.nn.CrossEntropyLoss(weight=loss_reweight)
    # metric_bce  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_reweight.unsqueeze(-1)) # C x 1
    # y is probability, not class tag, hence shall be floating point in [0,1]
    # x is of N x C or N x C x other
    if train_metric == "BCE":
        metric = torch.nn.BCEWithLogitsLoss()
    elif train_metric == "CE":
        metric = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"unknown metric: {train_metric}")

    ckpt_path = os.path.join(train_checkpoint_dir, train_tag)
    try:
        os.makedirs(ckpt_path, exist_ok=False)
    except FileExistsError:
        if train_clear_old_ckpt:
            import shutil
            import time
            for _ in range(3):
                logger.warning("YOU HAVE SPECIFIED train_clear_old_ckpt AND EXISTING CHECKPOINTS WERE FOUND.")
                logger.warning("YOU WILL HAVE 10s TO PRESS Ctrl-C BEFORE EVERYTHING GETS NUKED!")
                time.sleep(1)
            time.sleep(10)
            shutil.rmtree(ckpt_path)
            os.makedirs(ckpt_path, exist_ok=False)
        else:
            raise

    writer = SummaryWriter(comment=train_tag)

    with open(os.path.join(ckpt_path, f'params.json'), 'w') as fp:
        json.dump(cli_params, fp)

    np.savez(os.path.join(ckpt_path, f"random_state.npz"), train_idx=train_idx, test_idx=test_idx, selected_channels=dataset.selected_channels)

    if debug_use_profiler:
        logger.warning("profiler enabled")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name="./runs/profiler"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
    else:
        prof = None

    def train_epoch(epoch: int):
        progress_bar = tqdm(train_dataloader)
        losses = []
        all_labels = []
        all_probs = []
        all_predictions = []
        model.train()
        for i, (x, y) in enumerate(progress_bar):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            N, C, T = x.shape
            logits = model(x) # N, Classes, 1

            loss = metric(logits, y)
            loss.backward()
            optimizer.step()
            labels = torch.argmax(y, dim=-2).squeeze(-1)
            probs, predictions = torch.max(torch.sigmoid(logits).squeeze(-1), dim=-1)
            # torch.argmax(logits, dim=-2).squeeze()

            all_labels.append(labels)
            all_probs.append(probs)
            all_predictions.append(predictions)
            losses.append(loss)

            if i % train_show_loss_divisor == 0:
                acc = torch.mean((labels == predictions).to(torch.float32))
                progress_bar.set_postfix_str(f"loss={losses[-1].item():.4f}, acc={acc:.4f}")

            if debug_use_profiler:
                prof.step()

        labels = torch.cat(all_labels)
        probs = torch.cat(all_probs)
        predictions = torch.cat(all_predictions)
        acc = torch.mean((labels == predictions).to(torch.float32))
        # TODO: maybe use torch.mean
        epoch_mean_loss = np.mean(list(map(lambda x: x.item(), losses)))
        writer.add_pr_curve("train/pred", labels, probs, epoch)
        writer.add_scalar("train/loss", epoch_mean_loss, epoch)
        writer.add_scalar("train/acc", acc, epoch)
        logger.info("{} train: e={}, loss={:.4f}, acc={:.4f}", train_tag, e, epoch_mean_loss, acc)

        return epoch_mean_loss, acc

    @torch.no_grad()
    def eval_epoch(epoch: int):
        progress_bar = tqdm(test_dataloader)
        losses = []
        all_labels = []
        all_probs = []
        all_predictions = []
        model.eval()
        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)

            N, C, T = x.shape
            logits = model(x) # N, Classes, 1

            loss = metric(logits, y)

            labels = torch.argmax(y, dim=-2).squeeze(-1)
            probs, predictions = torch.max(torch.sigmoid(logits).squeeze(-1), dim=-1)
            acc = torch.mean((labels == predictions).to(torch.float32))

            all_labels.append(labels)
            all_probs.append(probs)
            all_predictions.append(predictions)
            losses.append(loss.item())

            progress_bar.set_postfix_str(f"loss={losses[-1]:.4f}, acc={acc:.4f}")

            if debug_use_profiler:
                prof.step()

        labels = torch.cat(all_labels)
        probs = torch.cat(all_probs)
        predictions = torch.cat(all_predictions)
        acc = torch.mean((labels == predictions).to(torch.float32))
        epoch_mean_loss = np.mean(losses)
        writer.add_pr_curve("eval/pred", labels, probs, epoch)
        writer.add_scalar("eval/loss", epoch_mean_loss, epoch)
        writer.add_scalar("eval/acc", acc, epoch)
        logger.info("{} eval: e={}, loss={:.4f}, acc={:.4f}", train_tag, epoch, epoch_mean_loss, acc)

        return epoch_mean_loss, acc

    if debug_use_profiler:
        prof.start()

    try:
        loss_rank = []
        acc_rank = []
        eval_loss_rank = []
        eval_acc_rank = []
        for e in range(train_epochs):
            if lr_sched is not None:
                lr = lr_sched.get_last_lr()
            else:
                lr = train_lr

            loss, acc = train_epoch(e)
            loss_rank.append((e, loss.item()))
            acc_rank.append((e, acc.item()))
            if e % train_eval_divisor == 0:
                loss, acc = eval_epoch(e)
                eval_loss_rank.append((e, loss.item()))
                eval_acc_rank.append((e, acc.item()))

            model_save_path = os.path.join(ckpt_path, f"{e}.pth")
            torch.save(model.state_dict(), model_save_path)

        loss_rank = sorted(loss_rank, key=lambda v: v[1])
        acc_rank = sorted(acc_rank, key=lambda v: v[1])
        eval_loss_rank = sorted(eval_loss_rank, key=lambda v: v[1])
        eval_acc_rank = sorted(eval_acc_rank, key=lambda v: v[1])

        writer.add_hparams(cli_params, {
            "best_train/best_acc": acc_rank[-1][1], "best_train/best_acc_epoch": acc_rank[-1][0], "best_train/best_loss": loss_rank[0][1], "best_train/best_loss_epoch": loss_rank[0][0],
            "best_eval/best_acc": eval_acc_rank[-1][1], "best_eval/best_acc_epoch": eval_acc_rank[-1][0], "best_eval/best_loss": eval_loss_rank[0][1], "best_eval/best_loss_epoch": eval_loss_rank[0][0]
        })
        logger.info("{} train: best_acc={} at Epoch {}, best_loss={} at Epoch {}", train_tag, acc_rank[-1][1], acc_rank[-1][0], loss_rank[0][1], loss_rank[0][0])
        logger.info("{} eval: best_acc={} at Epoch {}, best_loss={} at Epoch {}", train_tag, eval_acc_rank[-1][1], eval_acc_rank[-1][0], eval_loss_rank[0][1], eval_loss_rank[0][0])
        with open(os.path.join(ckpt_path, "best_info.json"), "w") as fp:
            json.dump(
                {
                    "loss_rank": loss_rank,
                    "acc_rank": acc_rank,
                    "eval_loss_rank": eval_loss_rank,
                    "eval_acc_rank": eval_acc_rank,
                },
                fp,
            )
    finally:
        writer.close()

        if debug_use_profiler:
            prof.stop()


if __name__ == "__main__":
    app()
