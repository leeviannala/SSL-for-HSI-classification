import torch
from decoder import DualNetwork, dual_loss
import argparse

def main(params):
    RUNS = 1
    MX_ITER = 1000000000
    SAMPLE_PERCENTAGE = params.SAMPLE_PERCENTAGE
    DATASET = params.DATASET
    DHCN_LAYERS = params.DHCN_LAYERS
    CONV_SIZE = params.CONV_SIZE
    H_MIRROR = params.H_MIRROR
    USE_HARD_LABELS = params.USE_HARD_LABELS
    LR = 1e-3
    model = DualNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=1e-4)
    loss = dual_loss()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1



def train_one_epoch(epoch_index, tb_writer, 
                    training_loader, optimizer, model,
                    loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--DHCN_LAYERS', default=1, type=int)
    parser.add_argument('--SAMPLE_PERCENTAGE', default=5, type=int)
    parser.add_argument('--DATASET', default="IndianPines", type=str)  # KSC, PaviaU, IndianPines, Botswana,    !!PaviaC
    parser.add_argument('--CONV_SIZE', default=3, type=int)  # 3,5,7
    parser.add_argument('--ROT', default=True, type=bool)  # False
    parser.add_argument('--MIRROR', default=True, type=bool)  # False
    parser.add_argument('--H_MIRROR', default='full', type=str)  # half, full
    parser.add_argument('--GPU', default='0,1,2,3', type=str)  # 0,1,2,3
    parser.add_argument('--ROT_N', default=1, type=int)  # False
    parser.add_argument('--MIRROR_N', default=1, type=int)  # False
    parser.add_argument('--USE_HARD_LABELS_N', default=1,type=int)

    # parser.add_argument('--RUNS', default=0, type=int) # False

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    params.ROT = True if params.ROT_N == 1 else False
    params.MIRROR = True if params.MIRROR_N == 1 else False
    params.USE_HARD_LABELS = True if params.USE_HARD_LABELS_N == 1 else False
    main(params)