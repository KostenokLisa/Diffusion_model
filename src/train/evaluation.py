import torch


def evaluation(test_loader, name=None, model_best=None, epoch=None, device='cpu'):
    with torch.no_grad():
        # EVALUATION
        if model_best is None:
            # load best performing model
            model_best = torch.load(name + '.model', map_location=device)

        model_best.eval()
        loss = 0.
        N = 0.
        for indx_batch, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(device)
            loss_t = model_best.forward(test_batch, reduction='sum')
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
        loss = loss / N

        if epoch is None:
            print(f'FINAL LOSS: nll={loss}')
        else:
            print(f'Epoch: {epoch}, val nll={loss}')

        return loss
