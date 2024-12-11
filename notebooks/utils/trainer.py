import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Trainer:
    def __init__(self, model, dataset, split_index=0):
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=1) #shuffle=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=50)
        self.criterion = nn.NLLLoss()

        self.train_mask = dataset.train_mask[:
                                             , split_index]
        self.val_mask = dataset.val_mask[:, split_index]
        self.test_mask = dataset.test_mask[:, split_index]

    def train(self):
        # Training loop
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.dataset.x, self.dataset.edge_index)
        loss = self.criterion(out[self.train_mask], self.dataset.y[self.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        # Evaluation loop
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.dataset.x, self.dataset.edge_index)
            pred = logits.argmax(dim=1)
            
            accs = []
            for mask in [self.train_mask, self.val_mask, self.test_mask]:
                acc = (pred[mask] == self.dataset.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
        return accs

    def run(self, num_epochs, print_freq=50):
        # Training process
        for epoch in range(num_epochs):
            loss = self.train()
            train_acc, val_acc, test_acc = self.evaluate()
            if epoch % print_freq == 0 or epoch == (num_epochs - 1):
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                    f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    def predict(self):
        batch = next(iter(self.loader))
        with torch.no_grad():
            logits, embeddings = self.model(batch.x, batch.edge_index, return_emb=True)

        predictions = logits.argmax(dim=1)

        return predictions, embeddings
    
    def get_metrics(self):
        batch = next(iter(self.loader))
        with torch.no_grad():
            logits, _ = self.model(batch.x, batch.edge_index, return_emb=True)
        probs = F.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)

        y_test = self.dataset.y[self.test_mask].cpu().numpy()
        y_pred = predictions[self.test_mask].cpu().numpy()
        probs = probs[self.test_mask].cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        if self.dataset.num_classes == 2:
            roc_auc = roc_auc_score(y_test, probs[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, probs, multi_class="ovo")
        
        return acc, precision, recall, f1, roc_auc
                
    def save_weights(self, path=None):
        if path is None:
            path = self._get_default_weights_path()

        # Save model
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path=None):
        if path is None:
            path = self._get_default_weights_path()

        self.model.load_state_dict(torch.load(path, weights_only=True))

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_default_weights_path(self):
        return f"models/weights/gcn_{self.dataset.name}.pth"
