
import os
import torch
import numpy as np
import argparse
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from tqdm import trange
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import FusionModule
from utils import seed_everything, setup_logger
from data import load_and_preprocess_data, create_loaders
import warnings
from umi_module import UMI
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")
import psutil
def use_cpus(gpus: list):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu*24, (gpu+1)*24)))
    p = psutil.Process()
    p.cpu_affinity(cpus)
    print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='UniMIX-AD')
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--modality', type=str, default='TMFP') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=16) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=16) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--mcg_loss_weight', type=float, default=0.1)  # Weight for MCG contrastive loss
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)

    return parser.parse_known_args()

def run_epoch(args, loader, encoder_dict, modality_dict, fusion_model, fami_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0, mcg_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    mcg_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed, batch_ori_observed in tqdm(loader):
        batch_samples = {k: v.cuda() for k, v in batch_samples.items()}
        batch_labels = batch_labels.cuda()
        batch_mcs = batch_mcs.cuda()
        batch_observed = batch_observed.cuda()
        batch_ori_observed = batch_ori_observed.cuda()

        fusion_input = []
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if modality == 'Text':
                encoded_samples = torch.zeros((samples.shape[0], 8, args.hidden_dim)).cuda()
            else:
                encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).cuda()
                #print(samples)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            fusion_input.append(encoded_samples)
        fusion_input = fami_model(fusion_input, batch_observed)

        outputs, gate_loss, mcg_loss = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())

            # Add gate loss and MCG loss to total loss
            # Ensure mcg_loss is scalar for multi-GPU compatibility
            # Handle both scalar and 1D tensor cases from DataParallel
            if mcg_loss.dim() == 0:
                mcg_loss_scalar = mcg_loss
            elif mcg_loss.dim() == 1:
                mcg_loss_scalar = mcg_loss.mean()  # Average across GPUs
            else:
                mcg_loss_scalar = mcg_loss.mean()  # Fallback for any other case
                
            # Handle gate loss (now potentially 1D tensor from DataParallel fix)
            gate_loss_scalar = gate_loss.mean() if gate_loss.dim() > 0 else gate_loss
            
            loss = task_loss + gate_loss_weight * gate_loss_scalar + mcg_loss_weight * mcg_loss_scalar
            gate_losses.append(gate_loss_scalar.item())
            mcg_losses.append(mcg_loss_scalar.item())
            
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())
        else:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses, mcg_losses, all_preds, all_labels, all_probs
    else:
        return all_preds, all_labels, all_probs


def train_and_evaluate(args, seed, save_path=None):
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)

    if args.data == 'adni':
        modality_dict = {'Text':0, 'MRI': 1, 'FDG': 2, 'AV45': 3}
        args.n_full_modalities = len(modality_dict)
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_data(args, modality_dict)
    train_loader, train_loader_shuffle, val_loader, test_loaders = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.use_common_ids)
    fusion_model = FusionModule(num_modalities, full_modality_index, args.num_patches, args.hidden_dim, n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout).cuda()
    fami_model = UMI(
        num_modalities=4,
        input_dims=[128, 128, 128, 128],
        sequence_lengths=[8, 16, 16, 16],
        d_model=128,
        target_length=16,
        num_heads=4,
        d_ff=256,
        dropout=0.1
    ).cuda()
    fami_model = torch.nn.DataParallel(fami_model)
    fusion_model = torch.nn.DataParallel(fusion_model)

    params = list(fami_model.parameters()) + list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().cuda() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).cuda())

    warmup_epochs = 15
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Cosine annealing scheduler
    total_epochs = args.train_epochs
    min_lr = 1e-6 
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
    criterion = torch.nn.CrossEntropyLoss().cuda() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).cuda())
    log_dir = 'logs/log_dir_seed_'+str(seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    best_val_acc = 0.0
    early_stop = 0

    if save_path is None:
        for epoch in trange(args.train_epochs):
            fusion_model.train()
            for encoder in encoder_dict.values():
                encoder.train()

            if epoch >= args.warm_up_epochs:
                train_loader_new = train_loader_shuffle
                warm_up_tag = ''
                train_epochs = args.train_epochs
            else:
                # activate modality-based sorting
                train_loader_new = train_loader
                warm_up_tag = 'Warm Up ' 
                train_epochs = args.warm_up_epochs

            ## Training
            task_losses, gate_losses, mcg_losses, train_preds, train_labels, train_probs = run_epoch(args, train_loader_new, encoder_dict, modality_dict, fusion_model, fami_model, criterion, device, is_training=True, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight, mcg_loss_weight=args.mcg_loss_weight)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            train_auc = roc_auc_score(train_labels, train_probs, multi_class='ovr')

            writer.add_scalar('loss/task_loss', np.array(task_losses).sum()/len(train_loader_new), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/gate_loss', np.array(gate_losses).sum()/len(train_loader_new), epoch)
            writer.add_scalar('train/mcg_loss', np.array(mcg_losses).sum()/len(train_loader_new), epoch)
            writer.add_scalar('train/Acc', train_acc, epoch)
            writer.add_scalar('train/F1 score', train_f1, epoch)
            writer.add_scalar('train/AUC', train_auc, epoch)

            print('Training')
            print(f"Epoch {epoch+1}/{train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f}, MCG Loss: {np.mean(mcg_losses):.2f} / Train Acc: {train_acc*100:.2f}, Train F1: {train_f1*100:.2f}, Train AUC: {train_auc*100:.2f}")
            ## Validation
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            with torch.no_grad():
                val_preds, val_labels, val_probs = run_epoch(args, val_loader, encoder_dict, modality_dict, fusion_model, fami_model, criterion, device)
            
            print('Prediction: ', )
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')
            writer.add_scalar('val/Acc', val_acc, epoch)
            writer.add_scalar('val/F1 score', val_f1, epoch)
            writer.add_scalar('val/AUC', val_auc, epoch)
            if val_acc > best_val_acc:
                print(f" [(**Best**) {warm_up_tag}Epoch {epoch+1}/{train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_model_fus = deepcopy(fusion_model)
                best_model_enc = deepcopy(encoder_dict)
                best_model_fami = deepcopy(fami_model)
                early_stop = 0

                # Save the best model
                if args.save:
                    os.makedirs('./saves', exist_ok=True)
                    save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
                    torch.save({
                        'fusion_model': best_model_fus.state_dict(),
                        'fami_model': best_model_fami.state_dict(),
                        'encoder_dict': {modality: deepcopy(encoder.state_dict()) for modality, encoder in best_model_enc.items()}
                    }, save_path)

                    print(f"Best model saved to {save_path}")
            else:
                early_stop += 1
            if early_stop == 40:
                print(f"Early stopping at epoch {epoch+1} with best val acc {best_val_acc*100:.2f}")
                break

            print(f"[Seed {seed}/{args.n_runs-1}] [{warm_up_tag}Epoch {epoch+1}/{train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f} / Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

    
    else:

        best_model_fus = fusion_model
        best_model_enc = encoder_dict
        # Load the saved model onto the correct device (GPU or CPU)
        checkpoint = torch.load(save_path, map_location=device)
        best_model_fus.load_state_dict(checkpoint['fusion_model'])
        best_model_fami.load_state_dict(checkpoint['fami_model'])
        for modality, encoder in best_model_enc.items():
            encoder.load_state_dict(checkpoint['encoder_dict'][modality])
            encoder.to(device)
            encoder.eval()

        best_model_fus.to(device)
        best_model_fami.to(device)

        best_val_acc = 0#accuracy_score(val_labels, val_preds)
        best_val_f1 = 0#f1_score(val_labels, val_preds, average='macro')
        best_val_auc = 0#roc_auc_score(val_labels, val_probs, multi_class='ovr')

    ## Test
    vis_text = []
    vis_mri = []
    vis_fdg = []
    vis_pet = []
    mean_accs = []
    mean_aucs = []
    mean_f1_scores = []
    #std_accs = []
    #std_aucs = []
    #std_f1_scores = []
    for mod in test_loaders:
        test_loader = test_loaders[mod]
        modalities_list = ['T', 'M', 'F', 'P']
        mod_mask = [1 if modality in mod else 0 for modality in modalities_list]
        print(f"Testing on {mod}")
        with torch.no_grad():
            test_preds, test_labels, test_probs = run_epoch(args, test_loader, best_model_enc, modality_dict, best_model_fus, best_model_fami, criterion, device)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        test_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
        vis_text.append(mod_mask[0])
        vis_mri.append(mod_mask[1])
        vis_fdg.append(mod_mask[2])
        vis_pet.append(mod_mask[3])
        mean_accs.append(test_acc)
        mean_aucs.append(test_auc)
        mean_f1_scores.append(test_f1)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    save_data = pd.DataFrame({'Text': vis_text, 'MRI': vis_mri, 'FDG': vis_fdg, 'PET': vis_pet, 'Accuracy': mean_accs, 'AUC': mean_aucs, 'F1': mean_f1_scores})
    save_data.to_excel(f'./results/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.xlsx', index=False)
    test_acc = 0
    test_f1 = 0
    test_auc = 0
    return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc

def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    #seeds = [6,7,8,9]
    val_accs = []

    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'FlexMoE',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        seed = int(seed)
        if (not args.save) & (args.load_model):
            save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
        else:
            save_path = None
        val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = train_and_evaluate(args, seed, save_path=save_path)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
    
    val_avg_acc = np.mean(val_accs)*100
    val_std_acc = np.std(val_accs)*100
    val_avg_f1 = np.mean(val_f1s)*100
    val_std_f1 = np.std(val_f1s)*100
    val_avg_auc = np.mean(val_aucs)*100
    val_std_auc = np.std(val_aucs)*100

    test_avg_acc = np.mean(test_accs)*100
    test_std_acc = np.std(test_accs)*100
    test_avg_f1 = np.mean(test_f1s)*100
    test_std_f1 = np.std(test_f1s)*100
    test_avg_auc = np.mean(test_aucs)*100
    test_std_auc = np.std(test_aucs)*100

    log_summary += f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} '
    log_summary += f'[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} '
    log_summary += f'[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / '  
    log_summary += f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} '
    log_summary += f'[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} '
    log_summary += f'[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} '  

    print(model_kwargs)
    print(f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}')
    print(f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}')

    logger.info(log_summary)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    use_cpus([5])
    main()