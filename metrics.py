
def calc_acc(output, target):
    predictions = output.argmax(dim=1, keepdim=True).squeeze()
    correct = (predictions == target).sum().item()
    accuracy = correct / target.size().numel()

    return accuracy
