from utils import AutoEncoder, Buffer, Config, reinit_encoder_weights, get_freqs
import torch
import wandb
import tqdm

cfg = Config()
encoder = AutoEncoder()
buffer = Buffer()

torch.set_grad_enabled(True)

try:
    print("Starting wandb")
    wandb.init(project="autoencoder", entity="armaanabraham-independent")
    print("Wandb started")
    num_batches = cfg.num_tokens // cfg.batch_size
    encoder_optim = torch.optim.Adam(
        encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )
    for i in tqdm.trange(num_batches):
        i = i % buffer.all_tokens.shape[0]
        acts = buffer.next()
        loss, feature_acts, l2_loss, l1_loss = encoder(acts)
        loss.backward()
        encoder.make_decoder_weights_and_grad_unit_norm()
        encoder_optim.step()
        encoder_optim.zero_grad()
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": l2_loss.item(),
            "l1_loss": l1_loss.item(),
        }
        del loss, feature_acts, l2_loss, l1_loss, acts
        if (i) % 100 == 0:
            wandb.log(loss_dict)
            print(loss_dict)
        if (i + 1) % 30000 == 0:
            encoder.save()
            freqs = get_freqs(encoder, buffer, 50)
            to_be_reset = freqs < 10 ** (-5.5)
            wandb.log({"reset_neurons": to_be_reset.sum().item()})
            reinit_encoder_weights(to_be_reset, encoder)
finally:
    encoder.save()
