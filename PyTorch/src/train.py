import torch
from .eval import evaluate

def train(model, train_dataloader, val_dataloader, epochs, is_classification=False):
    l1 = torch.nn.L1Loss()
    l2 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.transpose(0,1).unsqueeze(-1)
            optimizer.zero_grad()
            y_hat = model(x)
            if not is_classification:
                y = y.unsqueeze(-1)
            else:
                y = torch.nn.functional.one_hot(y, y_hat.shape[-1]).to(torch.float32)
            l1_loss = l1(y_hat, y)
            l2_loss = l2(y_hat, y)
            l2_loss.backward()
            optimizer.step()
        print(f"epoch {epoch}/{epochs}, train loss={l1_loss.item()}, train l2_loss={l2_loss.item()}")
        # evaluate(model, val_dataloader)
