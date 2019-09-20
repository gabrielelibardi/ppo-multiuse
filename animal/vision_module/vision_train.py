import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from vision_model import ImpalaCNNBase
from vision_dataset import DatasetVision, DatasetVisionRecurrent
from vision_functions import loss_func, plot_prediction


def vision_train(model, epochs, log_dir):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    # Get data
    dataset_train = DatasetVisionRecurrent("/home/abou/train_position_data.npz")
    dataset_test = DatasetVisionRecurrent("/home/abou/test_position_data.npz")

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": 128,
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_parameters)
    dataloader_test = DataLoader(dataset_test, **dataloader_parameters)

    device = torch.device("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    model.to(device)

    recurrent_hidden_states = torch.zeros(
        1, model.recurrent_hidden_state_size).to(device)

    for epoch in range(epochs):

        print('Epoch {}'.format(epoch))
        model.train()
        epoch_loss = 0
        t = tqdm.tqdm(dataloader_train)
        for idx, data in enumerate(t):

            if idx != 0:
                t.set_postfix(train_loss=avg_loss)

            images, pos, rot, masks = data

            images = images.to(device)
            pos = pos.to(device)
            rot = rot.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            pred_position, hx = model(
                masks=masks,
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = loss_func(
                pos, rot, pred_position[:, 0:2], pred_position[:, -1])

            train_loss = loss.item()
            epoch_loss += train_loss
            avg_loss = epoch_loss / (idx + 1)

            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.save("{}/model_{}.lol".format(log_dir, epoch), net_parameters)

        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            figure = plot_prediction(
                images, pos, rot, pred_position[:, 0:2], pred_position[:, -1])
            writer.add_figure(
                'train_figure_epoch_{}'.format(epoch), figure, epoch)

        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        epoch_loss = 0
        t = tqdm.tqdm(dataloader_test)
        for idx, data in enumerate(t):

            if idx != 0:
                t.set_postfix(test_loss=avg_loss)

            images, pos, rot, masks = data

            images = images.to(device)
            pos = pos.to(device)
            rot = rot.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            pred_position, hx = model(
                masks=masks,
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = loss_func(
                pos, rot, pred_position[:, 0:2], pred_position[:, -1])

            train_loss = loss.item()
            epoch_loss += train_loss
            avg_loss = epoch_loss / (idx + 1)

        if epoch % 10 == 0:
            figure = plot_prediction(
                images, pos, rot, pred_position[:, 0:2], pred_position[:, -1])
            writer.add_figure(
                'test_figure_epoch_{}'.format(epoch), figure, epoch)

        writer.add_scalar('test_loss', avg_loss, epoch)


if __name__ == "__main__":

    import os

    log_dir = "/home/abou/vision_module_logs"

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    net_parameters = {
        'num_inputs': 3,
        'recurrent': True,
        'hidden_size': 256,
        'image_size': 84
    }

    model = ImpalaCNNBase(**net_parameters)

    vision_train(model, 5000, log_dir)
