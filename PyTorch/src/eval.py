import torch
import numpy as np

def calc_MC(results) -> np.floating:
    x_mean = np.mean(np.array([x for x,y in results]))
    y_mean = np.mean(np.array([y for x,y in results]))
    covariance = np.mean(np.array([(x - x_mean)*(y-y_mean) for x,y in results]))
    variance_x = np.mean(np.array([np.power(x - x_mean, 2) for x,y in results]))
    variance_y = np.mean(np.array([np.power(y - y_mean, 2) for x,y in results]))
    result = covariance / (variance_x * variance_y)
    return result

def evaluate(model, dataloader, is_classification=False):
    model.eval()
    l1 = torch.nn.L1Loss()
    l2 = torch.nn.MSELoss()
    l1_losses = []
    l2_losses = []
    results = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.transpose(0,1).unsqueeze(-1)
            y_hat = model(x)
            if not is_classification:
                y = y.unsqueeze(-1)
            else:
                y = torch.nn.functional.one_hot(y, y_hat.shape[-1])
            l2_losses.append(l2(y, y_hat).item())
            l1_losses.append(l1(y, y_hat).item())
            results.append((y_hat, y))
    mc = calc_MC(results)
    print(f"Validation l1_loss = {np.array(l1_losses).mean()}, l2_loss = {np.array(l2_losses).mean()}, mc = {mc}")
