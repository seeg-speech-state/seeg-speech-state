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
    dataset_subject: Annotated[str, typer.Option()],
    train_tag: Annotated[str, typer.Option(help="Training tag, used for saving model")],
    train_batch_size: Annotated[int, typer.Option()],
    eval_tag: Annotated[str, typer.Option()],
    eval_use_epoch: Annotated[int, typer.Option()] = -1,
    eval_save_result: Annotated[bool, typer.Option()] = True,
    eval_use_split: Annotated[Literal['train', 'test', None], typer.Option()] = None,
    eval_report_digits: Annotated[int, typer.Option()] = 4,
    eval_analyze_grad: Annotated[bool, typer.Option()] = False,
    train_checkpoint_dir: Annotated[str, typer.Option()] = "./checkpoints",
    eval_output_dir: Annotated[str, typer.Option()] = "./evals",
    eval_clear_old_result: Annotated[bool, typer.Option()] = False,
    dataset_dir: Annotated[str, typer.Option()] = "./dataset",
    dataset_tag: Annotated[str | None, typer.Option()] = None,
    dataset_fast_dataloader: Annotated[bool, typer.Option()] = True
):
    cli_params = locals().copy()

    import torch
    import numpy as np
    from torch.utils.tensorboard.writer import SummaryWriter

    from dataset import Word300Dataset, collate_fn_time, SignalFilter
    from model import VanillaCNN1d, BatchNorm1d

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("infer.log")
    # logger.add("infer-info.log", level="INFO")

    assert torch.cuda.is_available(), "Must have CUDA"
    device = torch.device('cuda')
    dtype = torch.float32

    ckpt_path = os.path.join(train_checkpoint_dir, train_tag)

    with open(os.path.join(ckpt_path, "params.json"), "r") as fp:
        saved_params = json.load(fp)
        ksizes: list[int] = ast.literal_eval(saved_params['model_ksizes'])
        filters: list[int] = ast.literal_eval(saved_params['model_filters'])
        model_dropout = saved_params['model_dropout']
        train_metric = saved_params['train_metric']
        dataset_num_classes = saved_params['dataset_num_classes']
        dataset_channels = saved_params['dataset_channels']
        dataset_channel_config = saved_params['dataset_channel_config']
        dataset_cat_samples = saved_params['dataset_cat_samples']
        dataset_sig_filter_used = saved_params['dataset_sig_filter_used']
        dataset_channels_discard_ratio = saved_params['dataset_channels_discard_ratio']
        dataset_random_seed = saved_params['dataset_random_seed']

    dataset = Word300Dataset(Path(dataset_dir), dataset_subject, x_dtype=dtype, sig_filter_used=SignalFilter(dataset_sig_filter_used), channel_config=dataset_channel_config, tag=dataset_tag)
    logger.debug("using dataset={}", dataset)

    random_state = np.load(os.path.join(train_checkpoint_dir, train_tag, f'random_state.npz'))

    if dataset_channels_discard_ratio is not None and dataset_channels_discard_ratio > 0:
        logger.info("randomly discarding channels, ratio={}", dataset_channels_discard_ratio)
        this_time = dataset.random_discard_channels(dataset_channels_discard_ratio, dataset_random_seed)
        assert (this_time == random_state['selected_channels']).all()
    else:
        logger.debug("dataset_channels_discard_ratio not used")
        cli_params['dataset_channels_discard_ratio'] = None

    if eval_use_split is not None:
        if eval_use_split == "train":
            dataset = torch.utils.data.Subset(dataset, random_state['train_idx'])
        elif eval_use_split == "test":
            dataset = torch.utils.data.Subset(dataset, random_state['test_idx'])
        else:
            raise ValueError("Unknown eval_use_split={}", eval_use_split)
        logger.debug("loaded split={}, len(dataset)={}", eval_use_split, len(dataset))
    else:
        logger.warning("eval_use_split not specified, will load entire dataset")

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        batch_size=train_batch_size,
        collate_fn=collate_fn_time if dataset_cat_samples else None,
        shuffle=False,
        prefetch_factor=4 if dataset_fast_dataloader else None,
        persistent_workers=dataset_fast_dataloader,  # https://github.com/pytorch/pytorch/issues/47445
        drop_last=False,
    )

    if dataset_channels == -1:
        X, Y = dataset[0]
        dataset_channels = X.shape[0]
        logger.info("obtain dataset_channels={} from dataset", dataset_channels)

    output_layer = {"BCE": "sigmoid", "CE": "softmax"}[train_metric]

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
    logger.debug("using model={}", model)

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

    if eval_use_epoch == -1:
        with open(os.path.join(ckpt_path, "best_info.json"), 'r') as fp:
            d = json.load(fp)
            eval_use_epoch, eval_acc = d['eval_acc_rank'][-1]
        logger.info("loading model state, eval_use_epoch not specified, loading best epoch {} with eval_acc={}", eval_use_epoch, eval_acc)
    else:
        logger.info("loading model state, eval_use_epoch={}", eval_use_epoch)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, f'{eval_use_epoch}.pth'), weights_only=True))

    def eval_epoch(epoch: int):
        progress_bar = tqdm(test_dataloader)
        losses = []
        all_labels = []
        all_probs = []
        all_predictions = []
        all_grads = []
        with torch.enable_grad() if eval_analyze_grad else torch.no_grad():
            model.eval()
            for i, (x, y) in enumerate(progress_bar):
                x = x.to(device)
                if eval_analyze_grad:
                    x.requires_grad_()
                y = y.to(device)

                N, C, T = x.shape
                logits = model(x) # N, Classes, 1

                loss = metric(logits, y)

                labels = torch.argmax(y, dim=-2).squeeze(-1)
                probs, predictions = torch.max(torch.sigmoid(logits).squeeze(-1), dim=-1)
                acc = torch.mean((labels == predictions).to(torch.float32))

                if eval_analyze_grad:
                    grads = torch.autograd.grad(
                        inputs=x,
                        outputs=loss,
                        # retain_graph=True,
                        # create_graph=True
                    )[0]
                    # print(grads.shape)
                    all_grads.append(grads)

                all_labels.append(labels)
                all_probs.append(probs)
                all_predictions.append(predictions)
                losses.append(loss.item())

                progress_bar.set_postfix_str(f"loss={losses[-1]:.4f}, acc={acc:.4f}")

            labels = torch.cat(all_labels)
            probs = torch.cat(all_probs)
            predictions = torch.cat(all_predictions)
            acc = torch.mean((labels == predictions).to(torch.float32))
            epoch_mean_loss = np.mean(losses)
            # writer.add_pr_curve("eval/pred", labels, probs, epoch)
            # writer.add_scalar("eval/loss", epoch_mean_loss, epoch)
            # writer.add_scalar("eval/acc", acc, epoch)
            logger.info("{} eval: e={}, loss={:.4f}, acc={:.4f}", train_tag, epoch, epoch_mean_loss, acc)

            if eval_analyze_grad:
                # print(all_grads)
                grads = torch.cat(all_grads)
                return epoch_mean_loss, acc, labels, probs, predictions, grads
            else:
                return epoch_mean_loss, acc, labels, probs, predictions, None

    eval_path = os.path.join(eval_output_dir, eval_tag)
    try:
        os.makedirs(eval_path, exist_ok=False)
    except FileExistsError:
        if eval_clear_old_result:
            import shutil
            import time
            for _ in range(3):
                logger.warning("YOU HAVE SPECIFIED eval_clear_old_result AND EXISTING RESULTS WERE FOUND.")
                logger.warning("YOU WILL HAVE 10s TO PRESS Ctrl-C BEFORE EVERYTHING GETS NUKED!")
                time.sleep(1)
            time.sleep(10)
            shutil.rmtree(eval_path)
            os.makedirs(eval_path, exist_ok=False)
        else:
            raise

    with open(os.path.join(eval_path, f'params.json'), 'w') as fp:
        json.dump(cli_params, fp)

    rpt_txt_file = open(os.path.join(eval_path, "infer-report.txt"), "w")
    result_list = []
    try:
        loss, acc, labels, probs, predictions, grads = eval_epoch(eval_use_epoch)
        # torch.set_printoptions(profile="full")
        # print(labels)
        # print(probs)
        # print(predictions)

        from sklearn.metrics import classification_report
        y_true, y_pred = labels.numpy(force=True), predictions.numpy(force=True)
        rpt = classification_report(y_true, y_pred, target_names=['REST', 'SPEECH'], digits=eval_report_digits, output_dict=False)
        logger.info('\n' + str(rpt))
        rpt_txt_file.write(f"{train_tag}\n{str(rpt)}\n")

        rpt: dict = classification_report(y_true, y_pred, target_names=['REST', 'SPEECH'], output_dict=True)
        with open(os.path.join(eval_path, f'report.json'), 'w') as fp:
            rpt['train_tag'] = train_tag
            rpt['eval_tag'] = eval_tag
            rpt['subject'] = dataset_subject
            rpt['sig_filter'] = dataset_sig_filter_used
            rpt['channel_config'] = dataset_channel_config
            rpt['epoch'] = eval_use_epoch
            json.dump(rpt, fp)

        if eval_save_result:
            torch.save({
                "loss": loss,
                "acc": acc,
                "labels": labels,
                "probs": probs,
                "predictions": predictions,
                "grads": grads,
            }, os.path.join(eval_path, f'result.pth'))

    finally:
        # writer.close()
        rpt_txt_file.close()
        pass


if __name__ == "__main__":
    app()
