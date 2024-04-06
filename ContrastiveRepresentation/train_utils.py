import torch
import torch.nn.functional as F

def calculate_loss(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    return F.cross_entropy(y_logits, y).item()

def calculate_accuracy(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(y_logits, dim=1)
    correct = preds.eq(y).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def fit_contrastive_model(encoder, dataloader, num_iters, batch_size, learning_rate):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    encoder.train()
    losses = []

    for epoch in range(num_iters):
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_enc = encoder(anchor)
            positive_enc = encoder(positive)
            negative_enc = encoder(negative)
            loss = triplet_loss(anchor_enc, positive_enc, negative_enc)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


def fit_model(encoder: torch.nn.Module, classifier: Union[LinearModel, torch.nn.Module], train_loader, val_loader, args: Namespace):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
    encoder.train()
    classifier.train()
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(args.num_iters):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(ptu.device), y_batch.to(ptu.device)
            optimizer.zero_grad()
            with torch.no_grad():
                encoded = encoder(X_batch)
            y_logits = classifier(encoded)
            loss = F.cross_entropy(y_logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(calculate_accuracy(y_logits, y_batch))

        val_loss, val_acc = evaluate_model(encoder, classifier, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
    return train_losses, train_accs, val_losses, val_accs

def evaluate_model(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(ptu.device), y_batch.to(ptu.device)
            encoded = encoder(X_batch)
            y_logits = classifier(encoded)
            loss = F.cross_entropy(y_logits, y_batch)
            acc = calculate_accuracy(y_logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_acc += acc * X_batch.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_acc / len(data_loader.dataset)
    return avg_loss, avg_acc
