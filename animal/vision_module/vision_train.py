import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from vision_model import ImpalaCNNBase
from vision_dataset import DatasetVision, DatasetVisionRecurrent
from vision_functions import loss_func, plot_prediction


def vision_train(model, epochs, log_dir, train_data, test_data, batch_size=32):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    # Get data
    dataset_train = DatasetVisionRecurrent(train_data)
    dataset_test = DatasetVisionRecurrent(test_data)

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": batch_size,
        "drop_last": True
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_parameters)
    dataloader_test = DataLoader(dataset_test, **dataloader_parameters)

    device = torch.device("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    model.to(device)

    for epoch in range(epochs):

        print('Epoch {}'.format(epoch))
        model.train()
        epoch_loss = 0
        t = tqdm.tqdm(dataloader_train)
        for idx, data in enumerate(t):

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            if idx != 0:
                t.set_postfix(train_loss=avg_loss)

            images, pos, rot = data

            images = images.to(device)
            pos = pos.view(-1, 2).to(device)
            rot = rot.view(-1, 1).to(device)

            optimizer.zero_grad()
            pred_position, hx = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)
            pred_position = pred_position.view(-1, 3)

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
            images = images.view(
                -1, model.num_inputs, model.image_size, model.image_size)
            figure = plot_prediction(
                images, pos, rot, pred_position[:, 0:2], pred_position[:, -1])
            writer.add_figure(
                'train_figure_epoch_{}'.format(epoch), figure, epoch)

        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        model.train()
        epoch_loss = 0
        t = tqdm.tqdm(dataloader_test)
        for idx, data in enumerate(t):

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            if idx != 0:
                t.set_postfix(test_loss=avg_loss)

            images, pos, rot = data

            images = images.to(device)
            pos = pos.view(-1, 2).to(device)
            rot = rot.view(-1, 1).to(device)

            optimizer.zero_grad()
            pred_position, hx = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)
            pred_position = pred_position.view(-1, 3)

            loss = loss_func(
                pos, rot, pred_position[:, 0:2], pred_position[:, -1])

            train_loss = loss.item()
            epoch_loss += train_loss
            avg_loss = epoch_loss / (idx + 1)

        if epoch % 10 == 0:
            images = images.view(
                -1, model.num_inputs, model.image_size, model.image_size)
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

    vision_train(
        model, 5000, log_dir,
        train_data="/home/abou/train_position_data.npz",
        test_data="/home/abou/test_position_data.npz",
    )
