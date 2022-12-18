
def calc_acc(output, target):
    predictions = output.argmax(dim=1).squeeze()
    target = target.argmax(dim=1).squeeze()
    correct = (predictions == target).sum().item()
    accuracy = correct / target.size().numel()

    return accuracy
