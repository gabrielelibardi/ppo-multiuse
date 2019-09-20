import torch
import time
import tqdm
from tensorboardX import SummaryWriter
from torch.nn import ConstantPad2d
from torch.utils.data import DataLoader
from ppo.model import ImpalaCNNBase
from vision_dataset import DatasetVision
from vision_functions import loss_func


def vision_train(model, epochs, log_dir):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    # Get data
    dataset = DatasetVision(".../recordings.npz")

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": 128,
    }
    dataloader_train = DataLoader(dataset, **dataloader_parameters)

    device = torch.device("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    model.to(device)

    recurrent_hidden_states = torch.zeros(
        1, model.recurrent_hidden_state_size).to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        start = time.process_time()
        t = tqdm.tqdm(dataloader_train)
        for idx, data in enumerate(t):

            if idx != 0:
                t.set_postfix(train_loss=avg_loss)

            images, pos, rot, masks = data

            images = images.to(device)
            pos = pos.to(device)
            rot = rot.to(device)

            optimizer.zero_grad()
            pred_position, hx = model(
                masks=masks,
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            import ipdb; ipdb.set_trace()

            loss = loss_func(recon_images, images, mu, logvar)

            train_loss = loss.item()
            epoch_loss += train_loss
            avg_loss = epoch_loss / (idx + 1)

            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.save("{}/vae_{}.lol".format(log_dir, epoch), net_parameters)

        end = time.process_time()
        scheduler.step(avg_loss)

        print("Epoch[{}/{}] Loss: {:.3f} Time: {}".format(epoch+1,epochs, avg_loss,end-start))
        print("Learning rate {}".format(optimizer.param_groups[0]['lr']))

        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


if __name__ == "__main__":

    import os

    log_dir = "/home/abou/vae_logs"

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    net_parameters = {
        'num_inputs': 3,
        'recurrent': False,
        'hidden_size': 256,
        'image_size': 84
    }

    model = ImpalaCNNBase(**net_parameters)

    vision_train(model, 5000, log_dir)
