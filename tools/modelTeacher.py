
def train_step(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    losses = []
    train_loss = 0
    encoder.to(device)
    decoder.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        embeddings = encoder(X)
        X_hat = decoder(embeddings)

        # 2. Calculate loss
        loss = loss_fn(X_hat, X)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        losses.append(loss)

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.10f} ")
    return losses


def train_AE(trainloader, model: nn.Sequential, betta = 0.00001, epochs = 10, rand_seed = 42):
    optimizer = optim.Adam(model.parameters(),
                           lr=0.003,
                           weight_decay=betta)
    loss_fn = nn.MSELoss()
    torch.manual_seed(rand_seed)
    # Measure time
    train_time_start = timer()

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        losses = train_step(
            data_loader=trainloader,
            encoder=model[0],
            decoder=model[1],
            loss_fn=loss_fn,
            optimizer=optimizer
        )
    train_time_end = timer()
    print(f"Train time: {train_time_end - train_time_start}")
    return model