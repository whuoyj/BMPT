import numpy
import torch.distributed as dist
import torch
import clip
import os
import copy


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, scaler, max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")
    else:
        save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")


def load_checkpoint_fewshot(config, model, logger):
    if os.path.isfile(config.MODEL.FINETUNE_FEWSHOT):
        logger.info(f"==============> Resuming form {config.MODEL.FINETUNE_FEWSHOT}....................")
        checkpoint = torch.load(config.MODEL.FINETUNE_FEWSHOT, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        if "_orig_mod.module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.token_prefix"]

        if "_orig_mod.module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.token_suffix"]

        if "module.prompt_learner.neg_complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.neg_complete_text_embeddings"]
            
        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]
            
        if "_orig_mod.module.prompt_learner.neg_complete_text_embeddings" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.neg_complete_text_embeddings"]

        if "_orig_mod.module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.complete_text_embeddings"] 
        
        if not hasattr(model, 'module'):
            load_state_dict = {k.replace('module.', ''): v for k, v in load_state_dict.items()}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            if '_orig_mod.' in k:
                name = k.replace('_orig_mod.', '')
            else:
                name = k
            new_state_dict[name] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        # msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:

            logger.info(f"=> loaded successfully '{config.MODEL.FINETUNE_FEWSHOT}' (epoch {checkpoint['epoch']})")

            del checkpoint
            torch.cuda.empty_cache()

            return 0, 0
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def wise_state_dict(logger, ori_model, loaded_state_dict, weight_for_ft=None, keywords_to_exclude=None):
    """reference: https://github.com/mlfoundations/wise-ft"""
    if keywords_to_exclude is None:
        keywords_to_exclude = []
    finetuned_model = copy.deepcopy(ori_model)
    msg = finetuned_model.load_state_dict(loaded_state_dict, strict=False)
    logger.info(f'load finetuned model {msg}')

    state_dict_ori = dict(ori_model.named_parameters())
    state_dict_finetuned = dict(finetuned_model.named_parameters())
    # import pdb; pdb.set_trace();
    assert set(state_dict_ori) == set(state_dict_finetuned)

    fused_dict = {}
    for k in state_dict_ori:
        # Check if the current parameter name contains any of the excluded keywords
        if any(keyword in k for keyword in keywords_to_exclude):
            # If it does, use the finetuned model's parameters directly without fusion
            logger.info(f'weight fusion exception: {k}')
            fused_dict[k] = state_dict_finetuned[k]
        else:
            # Otherwise, fuse the weights according to the specified ratio
            fused_dict[k] = (1 - weight_for_ft) * state_dict_ori[k] + weight_for_ft * state_dict_finetuned[k]
    return fused_dict


def load_checkpoint(config, model, scaler, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME):
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        if "_orig_mod.module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.token_prefix"]

        if "_orig_mod.module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.token_suffix"]

        if "_orig_mod.module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["_orig_mod.module.prompt_learner.complete_text_embeddings"]

        if not hasattr(model, 'module'):
            load_state_dict = {k.replace('module.', ''): v for k, v in load_state_dict.items()}

        """reference: https://github.com/mlfoundations/wise-ft"""
        if config.TEST.WISE_FT != 0:

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in load_state_dict.items():
                if '_orig_mod.' in k:
                    k = k.replace('_orig_mod.', '')
                new_state_dict[k] = v

            print("Wise FT weight for fine-tuned:{}".format(config.TEST.WISE_FT))
            keywords_to_exclude = ['vision_reduce', 'text_reduce', 'vision_norm',
                                   'vision_head_downstream', 'vision_q_proj', 'vision_k_proj',
                                   'text_norm', 'logits_scale_vision']
            fused_state_dict = wise_state_dict(logger, ori_model=model, loaded_state_dict=new_state_dict,
                                               weight_for_ft=config.TEST.WISE_FT,
                                               keywords_to_exclude=keywords_to_exclude)
            msg = model.load_state_dict(fused_state_dict, strict=False)
            logger.info(f"Wise FT weight for fine-tuned model {config.TEST.WISE_FT}, fused model {msg}")
        else:
            logger.info(f"Without Wise FT weight for fine-tuned model!!")

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in load_state_dict.items():
                if '_orig_mod.' in k:
                    k = k.replace('_orig_mod.', '')
                new_state_dict[k] = v
            msg = model.load_state_dict(new_state_dict, strict=False)

            logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes


def is_main_process():
    return dist.get_rank() == 0


def gather_all_data(data):
    local_length = data.size(0)
    all_lengths = [torch.tensor(local_length, device="cuda") for _ in range(dist.get_world_size())]

    dist.all_gather(all_lengths, torch.tensor(local_length, device="cuda"))
    all_lengths = [l.item() for l in all_lengths]
    max_length = max(all_lengths)

    padded_data = torch.zeros((max_length, *data.shape[1:]), device=data.device, dtype=data.dtype)
    padded_data[:local_length] = data

    gathered_data = [torch.zeros_like(padded_data) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_data, padded_data)

    gathered_data = torch.cat([g[:l] for g, l in zip(gathered_data, all_lengths)], dim=0)
    return gathered_data


def calculate_topk(outputs, labels, topk=(1, 5)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    topk_accuracies = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        topk_accuracies.append(correct_k * 100.0 / batch_size)
    return topk_accuracies

def compute_model_stats(model, config, logger, batch_size=1, num_iterations=100):
    """
    Compute model statistics: parameters, GFLOPs, and throughput

    Args:
        model: The BDC-CLIP model
        config: Configuration object
        logger: Logger for output
        batch_size: Batch size for testing
        num_iterations: Number of iterations for throughput testing
    """
    import time
    import numpy as np

    logger.info("=" * 80)
    logger.info("COMPUTING MODEL STATISTICS")
    logger.info("=" * 80)

    # ========== 1. Compute Parameters ==========
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER STATISTICS")
    logger.info("=" * 80)

    total_params = 0
    trainable_params = 0

    param_groups = {
        'VPT Prompts': 0,
        'Cross-Modal Interaction': 0,
        'BDC Adapters': 0,
        'Text Encoder': 0,
        'Image Encoder': 0,
        'Other': 0
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

        # Categorize
        if 'VPT' in name:
            param_groups['VPT Prompts'] += num_params
        elif 'cross_modal' in name:
            param_groups['Cross-Modal Interaction'] += num_params
        elif any(k in name for k in ['vision_reduce', 'text_reduce', 'vision_q_proj', 'vision_k_proj',
                                       'vision_norm', 'text_norm', 'vision_bdc_head', 'BDC_Layer']):
            param_groups['BDC Adapters'] += num_params
        elif 'text_encoder' in name:
            param_groups['Text Encoder'] += num_params
        elif 'image_encoder' in name:
            param_groups['Image Encoder'] += num_params
        else:
            param_groups['Other'] += num_params

    logger.info(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    logger.info("\n" + "-" * 80)
    logger.info("Parameters by Module:")
    logger.info("-" * 80)
    for group, count in param_groups.items():
        if count > 0:
            logger.info(f"  {group:25s}: {count:12,} ({count/1e6:6.2f}M) - {100*count/total_params:5.2f}%")

    # ========== 2. Compute GFLOPs ==========
    logger.info("\n" + "=" * 80)
    logger.info("GFLOPS ESTIMATION")
    logger.info("=" * 80)

    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        num_frames = config.DATA.NUM_FRAMES
        input_size = 224  # CLIP default

        # Create dummy input: (batch, frames, channels, height, width)
        dummy_input = torch.randn(batch_size, num_frames, 3, input_size, input_size)

        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()

        model.eval()

        logger.info(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
            gflops = total_flops / 1e9
            gflops_per_sample = gflops / batch_size

        logger.info(f"Total GFLOPs (batch={batch_size}): {gflops:.2f}")
        logger.info(f"GFLOPs per sample: {gflops_per_sample:.2f}")

        # Detailed breakdown
        logger.info("\n" + "-" * 80)
        logger.info("GFLOPs Breakdown by Module:")
        logger.info("-" * 80)
        logger.info(flop_count_table(flops, max_depth=2))

    except ImportError:
        logger.info("Warning: fvcore not installed. Skipping GFLOPs computation.")
        logger.info("Install with: pip install fvcore")
        gflops = None
        gflops_per_sample = None
    except Exception as e:
        logger.info(f"Warning: Could not compute GFLOPs: {e}")
        gflops = None
        gflops_per_sample = None

    # ========== 3. Compute Throughput ==========
    logger.info("\n" + "=" * 80)
    logger.info("THROUGHPUT MEASUREMENT")
    logger.info("=" * 80)

    num_frames = config.DATA.NUM_FRAMES
    input_size = 224

    dummy_input = torch.randn(batch_size, num_frames, 3, input_size, input_size)

    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        logger.info(f"Device: GPU - {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device: CPU")

    model.eval()

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Iterations: {num_iterations}")

    # Warmup
    warmup_iterations = 10
    logger.info(f"\nWarming up ({warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    logger.info("Benchmarking...")
    times = []

    with torch.no_grad():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.time()
            _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.time()
            times.append(end - start)

    # Compute statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / mean_time

    logger.info("\n" + "-" * 80)
    logger.info("Throughput Results:")
    logger.info("-" * 80)
    logger.info(f"Mean inference time: {mean_time*1000:.2f} ms +/- {std_time*1000:.2f} ms")
    logger.info(f"Throughput (inference): {throughput:.2f} samples/sec")
    logger.info(f"Throughput (inference): {throughput*60:.2f} samples/min")

    # Estimate training throughput (typically 3-4x slower)
    training_factor = 3.5
    estimated_training_throughput = throughput / training_factor
    logger.info(f"Estimated training throughput: {estimated_training_throughput:.2f} samples/sec")

    # ========== Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nModel: BDC-CLIP ({config.MODEL.ARCH})")
    logger.info(f"Input: {batch_size} x {num_frames} x 3 x {input_size} x {input_size}")

    logger.info(f"\nParameters:")
    logger.info(f"  Total: {total_params/1e6:.2f}M")
    logger.info(f"  Trainable: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")

    if gflops is not None:
        logger.info(f"\nGFLOPs:")
        logger.info(f"  Per sample: {gflops_per_sample:.2f}")
        logger.info(f"  Batch ({batch_size}): {gflops:.2f}")

    logger.info(f"\nThroughput (batch={batch_size}):")
    logger.info(f"  Inference: {throughput:.2f} samples/sec ({mean_time*1000:.2f} ms/batch)")
    logger.info(f"  Training (estimated): {estimated_training_throughput:.2f} samples/sec")

    logger.info("=" * 80)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gflops': gflops,
        'gflops_per_sample': gflops_per_sample,
        'throughput': throughput,
        'mean_time': mean_time
    }