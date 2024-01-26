

for epoch in (range(EPOCHS)):
    loss, acc, auc = train_fn(model, train_dataloader, optimizer, criterion, device)
#     print("epoch - {}/{} train: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, auc))
    loss, acc, pre, rec, f1, auc = valid_fn(model, valid_dataloader, criterion, device)

    res = "epoch - {}/{} valid: - {:.3f} acc - {:.3f} pre - {:.3f} rec - {:.3f} f1 - {:3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, pre, rec, f1, auc)
    print(res)