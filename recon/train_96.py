import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from models import amplitude_loss
from dataset import Dataset, mel_spectrogram, get_dataset_filelist
import cosmos_tokenizer
from discrete_img import DiscreteImageTokenizer
from utils import AttrDict, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from vocos import Vocos
import shutil
torch.backends.cudnn.benchmark = True

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from CosmosTokenizer.cosmos_tokenizer.modules import DecoderType, DiscreteQuantizer, EncoderType

params = dict(
    attn_resolutions=[6, 12],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=1,
    spatial_compression=8,
    num_res_blocks=2,
    out_channels=1,
    resolution=96,
    patch_size=2,
    patch_method="haar",
    z_channels=256,
    z_factor=2,
    quantizer=DiscreteQuantizer.VQ.name,
    embedding_dim=64,
    num_embeddings=8192,
    num_quantizers=1,
    name="DI",
    encoder=EncoderType.Default.name,
    decoder=DecoderType.Default.name,
)

def train(h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(0))
    model = DiscreteImageTokenizer(**params).to(device)

    print("Model: ")
    print(model)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_model = scan_checkpoint(h.checkpoint_path, 'vae_')


    steps = 0
    if cp_model is None:
        state_dict_vae = None
        last_epoch = -1
    else:
        state_dict_vae = load_checkpoint(cp_model, device)
        model.load_state_dict(state_dict_vae['encoder'])
        steps = 0
        last_epoch = -1

    optim_g = torch.optim.AdamW(itertools.chain(model.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_vae is not None:
        model.load_state_dict(state_dict_vae['encoder'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    
    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                       h.hop_length, h.sampling_rate, shuffle=True, device=device, train=True)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validset = Dataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                       h.hop_length, h.sampling_rate, shuffle=False, device=device, train=False)
    
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    #model = model.to(dtype=torch.bfloat16)
    model.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):

        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            y_mel = batch
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            #y_mel = y_mel.to(dtype=torch.bfloat16)
            out_train = model(y_mel)
            y_g_mel = out_train["reconstructions"]
            # Generator
            optim_g.zero_grad()
            # Losses defined on log mel spectra
            L_M = F.l1_loss(y_mel, y_g_mel)*5.0
            Mel_L2_error = amplitude_loss(y_mel, y_g_mel)*25.0
            quant_loss = out_train["quant_loss"].mean()
            L_G =  L_M + quant_loss*0.25
            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    Mel_error = (F.l1_loss(y_mel, y_g_mel)*5.0).item()
                    Mel_L2_error = (amplitude_loss(y_mel, y_g_mel)*25.0).item()
                    quant_loss = quant_loss.item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, Mel Spectrogram L2 Loss : {:4.3f}, Quant Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, L_G,  Mel_error, Mel_L2_error, quant_loss, time.time() - start_b))

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/vae_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'encoder': model.state_dict(),
                                'steps': steps,
                                 'epoch': epoch})

            # Tensorboard summary logging
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0:  # and steps != 0:
                model.eval()
                torch.cuda.empty_cache()
                val_Mel_err_tot = 0
                val_Mel_L2_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        y_mel = batch
                        #y_mel = y_mel.to(dtype=torch.bfloat16)
                        out_eval = model(y_mel.to(device))
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))                        
                        val_Mel_err_tot += (F.l1_loss(y_mel, out_eval.reconstructions)*5.0).item()
                        val_Mel_L2_err_tot += (amplitude_loss(y_mel, out_eval.reconstructions)*25.0).item()


                        if j <= 4:
                            if steps == 0:
                                y_plot_tensor = y_mel[0, 0] * 5.0
                                y_plot = y_plot_tensor.cpu().float().numpy()  # 再转 numpy
                                sw.add_figure('gt/y_mel_{}'.format(j), plot_spectrogram(y_plot), steps)

                            y_plot_tensor_g = y_g_mel[0, 0] * 5.0
                            y_plot_g = y_plot_tensor_g.cpu().float().numpy()
                            sw.add_figure('generated/y_g_mel_{}'.format(j), plot_spectrogram(y_plot_g), steps)

                    val_Mel_err = val_Mel_err_tot / (j+1)
                    val_Mel_L2_err = val_Mel_L2_err_tot / (j+1)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_L2_loss", val_Mel_L2_err, steps)

                model.train()

            steps += 1

        scheduler_g.step()
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    config_file = 'config_96.json'

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, 'config_96.json', h.checkpoint_path)
    
    src = "train_96.py"
    dst_dir = h.checkpoint_path
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "train_96.py")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} 不存在！")
    shutil.copyfile(src, dst)
    print(f"已将 {src} 复制到 {dst}")
    
    
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == '__main__':
    main()

